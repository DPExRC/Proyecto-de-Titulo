# =============================================================================
# serial_bridge.py — Puente serie: muestras crudas → DSP → features → ángulos
# =============================================================================
# Lee tramas crudas del Arduino ("S,<v_biceps>,<v_triceps>,<v_antebrazo>"),
# aplica el pipeline DSP completo en Python (filtro IIR + ventaneo +
# extracción de features), deposita vectores de 12 features en una cola
# para que el hilo de control ejecute la inferencia del regresor, y envía
# los dos ángulos calculados de vuelta al Arduino.
#
# Protocolo serial:
#   Arduino → PC:  "S,<adc_biceps>,<adc_triceps>,<adc_antebrazo>\n"
#                   valores enteros 0–1023 (ADC de 10 bits)
#   PC → Arduino:  "A,<angulo_codo>,<angulo_muneca>\n"
#                   valores float con 1 decimal, p. ej. "A,72.3,15.0\n"
#
# NOTA: el firmware Arduino (emg_v3.ino) debe actualizarse para:
#   1. Emitir tramas "S,..." en vez del formato anterior de features.
#   2. Parsear comandos "A,<codo>,<muneca>" con dos ángulos en vez de uno.
#   Hasta que eso ocurra, este módulo no puede usarse con hardware real.
# =============================================================================

import os
import sys

# Ajuste dinámico de sys.path para que detecte la raíz del proyecto 'Servos'
# os.path.dirname(__file__) es 'src/core'
# El primer '..' sube a 'src', el segundo '..' sube a 'Servos'
root_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if root_path not in sys.path:
    sys.path.insert(0, root_path)

import serial
import threading
import queue
import time

# Ahora las importaciones absolutas desde la raíz funcionarán correctamente
from src.config import BAUDRATE, NOMBRES_CANALES
from src.processing.dsp import CapturadorVentanas
from src.processing.calibration import CalibradorEMG

PREFIJO_MUESTRA = "S,"   # trama entrante: "S,<v0>,<v1>,<v2>"
PREFIJO_ANGULO  = "A,"   # comando saliente: "A,<codo>,<muneca>"
N_CANALES_ESPERADOS = len(NOMBRES_CANALES)


class SerialBridge:
    """Gestiona la conexión serie bidireccional con el Arduino.

    El hilo de lectura (leer_muestras) aplica el pipeline DSP completo
    en Python y deposita vectores de 12 features en la cola provista.
    El hilo de control escribe ángulos mediante enviar_angulos(), que usa
    un lock interno para evitar colisiones entre lecturas y escrituras
    concurrentes sobre el mismo puerto serie.
    """

    def __init__(self, port: str, baudrate: int = BAUDRATE):
        self.port      = port
        self.baudrate  = baudrate
        self.ser: serial.Serial | None = None
        self._lock_write = threading.Lock()
        self.calibrador = CalibradorEMG()

    def ejecutar_calibracion(self, duracion_reposo_s: float = 3.0,
                              duracion_mvc_s: float = 3.0) -> bool:
        """Realiza la calibración baseline/MVC en el puerto serie abierto."""
        if not self.ser or not self.ser.is_open:
            print("[SerialBridge] No hay conexión serie activa para calibrar.")
            return False

        capturador = CapturadorVentanas()
        self.calibrador.ejecutar(self.ser, capturador,
                                 duracion_reposo_s, duracion_mvc_s)
        return self.calibrador.calibrado

    # ------------------------------------------------------------------
    def conectar(self) -> bool:
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=2.0)
            time.sleep(2.0)
            self.ser.reset_input_buffer()
            print(f"[SerialBridge] Conectado: {self.port} @ {self.baudrate} baud")
            return True
        except serial.SerialException as e:
            print(f"[SerialBridge] Error al conectar en {self.port}: {e}")
            return False

    def desconectar(self):
        if self.ser and self.ser.is_open:
            self.ser.close()
            print("[SerialBridge] Desconectado.")

    # ------------------------------------------------------------------
    def enviar_angulos(self, angulo_codo: float, angulo_muneca: float):
        """Envía ambos ángulos objetivo al Arduino.
        Protocolo: "A,<angulo_codo>,<angulo_muneca>\\n"
        Thread-safe: usa lock interno para no colisionar con lecturas."""
        cmd = f"{PREFIJO_ANGULO}{angulo_codo:.1f},{angulo_muneca:.1f}\n"
        with self._lock_write:
            try:
                if self.ser and self.ser.is_open:
                    self.ser.write(cmd.encode())
            except serial.SerialException as e:
                print(f"[SerialBridge] Error al enviar ángulos: {e}")

    # ------------------------------------------------------------------
    def leer_muestras(self, cola_features: queue.Queue,
                       flag_activo: threading.Event):
        """Thread target: lee tramas crudas, aplica DSP y deposita
        vectores de features en cola_features.

        Si la cola está llena (el hilo de control no da abasto), descarta
        el vector más antiguo para mantener la latencia baja — es
        preferible perder una actualización antigua que acumular retraso.
        """
        capturador = CapturadorVentanas()

        while flag_activo.is_set():
            if not self.ser or not self.ser.is_open:
                time.sleep(0.05)
                continue

            try:
                raw = self.ser.readline()
            except serial.SerialException as e:
                print(f"[SerialBridge] Error de lectura: {e}")
                time.sleep(0.1)
                continue

            if not raw:
                continue

            linea = raw.decode("utf-8", errors="ignore").strip()

            # Mensajes de diagnóstico del firmware
            if linea.startswith("#"):
                print(f"[firmware] {linea}")
                continue

            if not linea.startswith(PREFIJO_MUESTRA):
                continue

            partes = linea[len(PREFIJO_MUESTRA):].split(",")
            if len(partes) != N_CANALES_ESPERADOS:
                continue

            try:
                valores = [float(p) for p in partes]
            except ValueError:
                continue

            vector = capturador.procesar_trama(valores)
            if vector is None:
                continue

            vector = self.calibrador.normalizar(vector)

            # Depositar en cola, descartando el más antiguo si está llena
            if cola_features.full():
                try:
                    cola_features.get_nowait()
                except queue.Empty:
                    pass

            try:
                cola_features.put_nowait(vector)
            except queue.Full:
                pass   # descarte silencioso — preferible a bloquear el hilo