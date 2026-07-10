# =============================================================================
# serial_bridge.py — Puente serie robusto + integración DSP/calibración
# =============================================================================
# Capa de TRANSPORTE (conexión, reconexión automática, lectura/escritura
# segura) + capa de APLICACIÓN (parseo de tramas "S,...", extracción de
# features vía CapturadorVentanas, normalización vía CalibradorEMG).
#
# Protocolo (debe coincidir EXACTO con emg_bridge_v4.ino):
#   RX (Arduino -> PC):  "S,<adc_biceps>,<adc_triceps>,<adc_antebrazo>\n"
#                        "A\n" (ACK echo para loopback E2E)
#   TX (PC -> Arduino):  "A,<angulo_codo>,<angulo_muneca>\n"
#
# CAMBIO respecto a la versión anterior: enviar_angulos() ahora antepone
# el prefijo "A," — sin él, el firmware descarta la trama silenciosamente
# (ver procesarLinea() en el .ino: exige linea[0]=='A' y linea[1]==',').
# 
# NUEVO: enviar_angulos_con_medicion() espera el ACK del firmware para
# medir latencia electrónica E2E real (τ_elec = t_echo - t_trigger).
# =============================================================================

import os
import sys
import time
import queue
import threading
import logging
from typing import Optional
import serial

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import NOMBRES_CANALES
from src.processing.dsp import CapturadorVentanas
from src.processing.calibration import CalibradorEMG, RUTA_CALIBRACION_DEFAULT

PROTOCOLO_PREFIJO_RX = "S,"   # Arduino -> PC (muestras crudas)
PROTOCOLO_PREFIJO_TX = "A,"   # PC -> Arduino (ángulos objetivo)
PROTOCOLO_ACK = "A"            # Arduino -> PC (echo de comando procesado)

# Latencia mecánica nominal de los servos KS-3518 (para recorrido típico de 60°)
LATENCIA_MECANICA_MS = 120.0

LOG_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_PATH = os.path.join(LOG_DIR, "serial_bridge.log")

logger = logging.getLogger("servos.serial_bridge")
if not logger.handlers:
    handler = logging.FileHandler(LOG_PATH)
    handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


def parsear_trama_emg(linea: str) -> Optional[list[float]]:
    """Parsea una trama EMG cruda sin depender del puerto serie."""
    if not linea.startswith(PROTOCOLO_PREFIJO_RX):
        return None

    partes = linea[len(PROTOCOLO_PREFIJO_RX):].split(",")
    if len(partes) != len(NOMBRES_CANALES):
        return None

    try:
        return [float(p) for p in partes]
    except ValueError:
        return None


def procesar_trama_emg(valores_crudos: list[float],
                       capturador: CapturadorVentanas,
                       calibrador: CalibradorEMG) -> Optional[list[float]]:
    """Procesa una trama de valores crudos y devuelve features normalizadas."""
    vector = capturador.procesar_trama(valores_crudos)
    if vector is None:
        return None

    return calibrador.normalizar(vector)


class SerialBridge:
    def __init__(self, port="COM3", baudrate=115200, timeout=1.0):
        """
        Inicializa el puente serial. NO conecta automáticamente en el
        constructor — el llamador decide cuándo conectar con conectar(),
        para evitar aperturas dobles del puerto (cada apertura fuerza un
        reset del Arduino y una espera de ~2s).
        """
        self.port = port
        self.baudrate = baudrate
        self.timeout = timeout
        self.ser = None
        self.conectado = False
        self.lock = threading.Lock()  # Previene que lectura y escritura choquen

        self.capturador = CapturadorVentanas()
        self.calibrador = CalibradorEMG()
        self.logger = logger

    # ------------------------------------------------------------------
    def conectar(self) -> bool:
        """Intenta establecer conexión con el Arduino de forma segura.
        Retorna True si la conexión quedó lista, False si falló."""
        try:
            if self.ser is not None and self.ser.is_open:
                self.ser.close()

            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)

            time.sleep(2)  # Pausa necesaria para que el Arduino haga su auto-reset

            # --- Limpieza de Buffer (Flush) inicial ---
            self.ser.reset_input_buffer()
            self.ser.reset_output_buffer()

            self.conectado = True
            self.logger.info("Conectado a %s a %s baudios", self.port, self.baudrate)
            return True

        except serial.SerialException as e:
            self.conectado = False
            self.logger.error("Error de conexión en %s: %s", self.port, e)
            return False

    # ------------------------------------------------------------------
    def leer_trama(self):
        """Lee una línea del serial con manejo de tramas corruptas y
        desconexiones. Retorna el string crudo (ej. 'S,512,480,300') o
        None si no hay dato válido en este intento."""
        if not self.conectado or self.ser is None:
            time.sleep(1)
            self.conectar()  # reconexión automática
            return None

        try:
            linea = self.ser.readline()

            if not linea:  # timeout sin datos
                return None

            linea_decodificada = linea.decode('ascii', errors='ignore').strip()

            if not linea_decodificada:
                return None

            return linea_decodificada

        except serial.SerialException:
            self.logger.warning("Conexión perdida repentinamente. Intentando reconectar...")
            self.conectado = False
            if self.ser:
                self.ser.close()
            return None

        except Exception as e:
            self.logger.exception("Error inesperado decodificando datos: %s", e)
            return None

    # ------------------------------------------------------------------
    def _parsear_trama_emg(self, linea: str):
        """Wrapper de compatibilidad para la función pura parsear_trama_emg()."""
        return parsear_trama_emg(linea)

    def leer_muestras(self, cola_features: queue.Queue, flag_activo: threading.Event):
        """Loop pensado para correr en su propio hilo (t_serial en
        main.py). Lee tramas continuamente, las pasa por el pipeline
        DSP (CapturadorVentanas) y, cuando se completa un vector de 12
        features, lo normaliza con la calibración activa y lo deposita
        en `cola_features` para que hilo_control lo consuma."""
        while flag_activo.is_set():
            linea = self.leer_trama()
            if linea is None:
                continue

            valores = self._parsear_trama_emg(linea)
            if valores is None:
                continue  # comentario '#', 'READY', o trama corrupta — se ignora

            vector_normalizado = procesar_trama_emg(valores, self.capturador, self.calibrador)
            if vector_normalizado is None:
                continue  # ventana aún no completa / no tocaba emitir todavía

            try:
                cola_features.put(vector_normalizado, block=False)
            except queue.Full:
                # Si el hilo de control se atrasó, se descarta la muestra
                # más vieja pendiente de la cola y se mete la nueva —
                # preferible a acumular latencia en el control del brazo.
                try:
                    cola_features.get_nowait()
                except queue.Empty:
                    pass
                cola_features.put_nowait(vector_normalizado)

    # ------------------------------------------------------------------
    def enviar_angulos(self, angulo_codo, angulo_muneca) -> bool:
        """Envía los ángulos al Arduino de forma segura, con el prefijo
        'A,' que exige el firmware (ver procesarLinea() en el .ino)."""
        if not self.conectado or self.ser is None:
            return False

        trama_salida = f"{PROTOCOLO_PREFIJO_TX}{angulo_codo},{angulo_muneca}\n"

        with self.lock:
            try:
                self.ser.write(trama_salida.encode('ascii'))
                self.ser.flush()
                return True
            except serial.SerialException:
                self.conectado = False
                self.logger.error("Error enviando comandos al actuador")
                return False

    # ------------------------------------------------------------------
    def enviar_angulos_con_medicion(self, angulo_codo, angulo_muneca, 
                                     timeout_ack=0.5) -> dict:
        """
        Envía ángulos y espera el ACK del firmware para medir latencia E2E real.
        
        Retorna un diccionario con:
        - "exito": bool — si se envió y recibió ACK correctamente
        - "latencia_electronica_ms": float — tiempo t_echo - t_trigger
        - "latencia_e2e_ms": float — latencia electrónica + mecánica nominal
        - "angulo_codo": float — valor enviado
        - "angulo_muneca": float — valor enviado
        
        Protocolo:
        1. Registra t_trigger con time.perf_counter() de alta resolución
        2. Envía trama "A,codo,muneca\n"
        3. Espera ACK "A\n" desde el firmware (bloqueante con timeout)
        4. Registra t_echo al recibir ACK
        5. Calcula latencia = t_echo - t_trigger
        """
        if not self.conectado or self.ser is None:
            return {
                "exito": False,
                "latencia_electronica_ms": None,
                "latencia_e2e_ms": None,
                "angulo_codo": angulo_codo,
                "angulo_muneca": angulo_muneca,
            }

        trama_salida = f"{PROTOCOLO_PREFIJO_TX}{angulo_codo},{angulo_muneca}\n"
        t_trigger = time.perf_counter()

        with self.lock:
            try:
                # 1. Enviar comando
                self.ser.write(trama_salida.encode('ascii'))
                self.ser.flush()

                # 2. Esperar ACK con timeout
                # (En el Arduino, el ACK se envía en el próximo ciclo de control,
                # típicamente dentro de ~20 ms)
                timeout_antiguo = self.ser.timeout
                self.ser.timeout = timeout_ack

                try:
                    ack_linea = self.ser.readline()
                    t_echo = time.perf_counter()
                    ack_decodificado = ack_linea.decode('ascii', errors='ignore').strip()
                    
                    if ack_decodificado == PROTOCOLO_ACK:
                        latencia_elec_ms = (t_echo - t_trigger) * 1000.0
                        latencia_e2e_ms = latencia_elec_ms + LATENCIA_MECANICA_MS
                        
                        self.logger.info(
                            "Latencia E2E medida: τ_elec=%.2f ms, τ_mech=%.1f ms → "
                            "E2E=%.1f ms",
                            latencia_elec_ms, LATENCIA_MECANICA_MS, latencia_e2e_ms
                        )
                        
                        return {
                            "exito": True,
                            "latencia_electronica_ms": latencia_elec_ms,
                            "latencia_e2e_ms": latencia_e2e_ms,
                            "angulo_codo": angulo_codo,
                            "angulo_muneca": angulo_muneca,
                        }
                    else:
                        self.logger.warning(
                            "ACK recibido pero contenido inesperado: %s",
                            repr(ack_decodificado)
                        )
                        return {
                            "exito": False,
                            "latencia_electronica_ms": None,
                            "latencia_e2e_ms": None,
                            "angulo_codo": angulo_codo,
                            "angulo_muneca": angulo_muneca,
                        }

                except serial.SerialTimeoutException:
                    self.logger.warning(
                        "Timeout esperando ACK del firmware (timeout=%.2f s)",
                        timeout_ack
                    )
                    return {
                        "exito": False,
                        "latencia_electronica_ms": None,
                        "latencia_e2e_ms": None,
                        "angulo_codo": angulo_codo,
                        "angulo_muneca": angulo_muneca,
                    }

                finally:
                    self.ser.timeout = timeout_antiguo

            except serial.SerialException as e:
                self.conectado = False
                self.logger.error("Error enviando comandos con medición: %s", e)
                return {
                    "exito": False,
                    "latencia_electronica_ms": None,
                    "latencia_e2e_ms": None,
                    "angulo_codo": angulo_codo,
                    "angulo_muneca": angulo_muneca,
                }

    # ------------------------------------------------------------------
    def ejecutar_calibracion(self, duracion_reposo_s: float = 3.0,
                              duracion_mvc_s: float = 3.0) -> bool:
        """Corre las 4 fases de calibración (baseline + MVC x3 canales)
        sobre la conexión activa, y guarda el resultado en disco. Usa un
        CapturadorVentanas separado del de leer_muestras() para no
        interferir con el estado de ventaneo de producción."""
        if not self.conectado or self.ser is None:
            self.logger.warning("No se puede calibrar sin conexión activa")
            return False

        capturador_calib = CapturadorVentanas()
        with self.lock:
            self.calibrador.ejecutar(self.ser, capturador_calib,
                                      duracion_reposo_s, duracion_mvc_s)
        self.calibrador.guardar(RUTA_CALIBRACION_DEFAULT)
        return self.calibrador.calibrado

    def cargar_calibracion(self, ruta: str = RUTA_CALIBRACION_DEFAULT) -> bool:
        """Carga una calibración previamente guardada, sin necesidad de
        repetir las 4 fases interactivas."""
        return self.calibrador.cargar(ruta)

    # ------------------------------------------------------------------
    def desconectar(self):
        """Cierra el puerto de forma limpia y segura."""
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.conectado = False
        self.logger.info("Puerto cerrado correctamente")

    # Alias por compatibilidad con código que use el nombre anterior
    cerrar = desconectar
