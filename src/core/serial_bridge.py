# =============================================================================
# serial_bridge.py — Comunicación serial con firmware EMG v3.0
# =============================================================================
# Protocolo RX (desde Arduino):
#   READY                         → arranque del firmware
#   F,<rms0>,<zcr0>,<rms1>,<zcr1> → vector de features (modo offload)
#   # ...                         → líneas de log/diagnóstico (ignorar en ML)
#
# Protocolo TX (hacia Arduino):
#   C,<0|1|2>\n                   → enviar clase predicha
#   A,<angulo>\n                  → control directo de ángulo
# =============================================================================

import serial
import threading
import time
import queue
import sys

class SerialBridge:
    """Gestiona la comunicación serial con el firmware EMG v3.0."""

    def __init__(self, port: str, baudrate: int = 115200, timeout: float = 2.0):
        self.port     = port
        self.baudrate = baudrate
        self.timeout  = timeout
        self.ser      = None
        self._lock    = threading.Lock()
        self._activo  = False

    # ------------------------------------------------------------------
    def conectar(self) -> bool:
        """Abre el puerto serial y espera la señal READY del firmware."""
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=self.timeout)
            time.sleep(2.0)   # espera reset del Arduino al abrir puerto
            self.ser.reset_input_buffer()

            t0 = time.time()
            while time.time() - t0 < 30.0:
                linea = self.ser.readline().decode("utf-8", errors="ignore").strip()
                if linea == "READY":
                    self._activo = True
                    print(f"[SerialBridge] Conectado en {self.port}")
                    return True
                if linea:
                    print(f"[firmware] {linea}")

            print("[SerialBridge] Tiempo de espera agotado — READY no recibido")
            return False

        except serial.SerialException as e:
            print(f"[SerialBridge] Error al conectar: {e}")
            return False

    # ------------------------------------------------------------------
    def desconectar(self):
        """Envía servo a posición segura y cierra el puerto."""
        if self.ser and self.ser.is_open:
            try:
                self.enviar_clase(0)   # REPOSO antes de cerrar
                time.sleep(0.1)
                self.ser.close()
            except Exception:
                pass
        self._activo = False
        print("[SerialBridge] Puerto cerrado")

    # ------------------------------------------------------------------
    def leer_features(self, cola: queue.Queue, flag_activo: threading.Event):
        """
        Hilo de lectura continua. Parsea líneas F,... y deposita vectores
        de 4 features en la cola. Las líneas de log (#) se imprimen al
        stderr sin pasar al pipeline ML.
        """
        while flag_activo.is_set() and self.ser and self.ser.is_open:
            try:
                raw = self.ser.readline()
                if not raw:
                    continue
                linea = raw.decode("utf-8", errors="ignore").strip()

                if linea.startswith("#"):
                    print(f"[firmware] {linea}", file=sys.stderr)

                elif linea.startswith("F,"):
                    partes = linea[2:].split(",")
                    if len(partes) == 4:
                        try:
                            features = [float(p) for p in partes]
                            # Drop-and-replace si la cola está llena
                            if cola.full():
                                try:
                                    cola.get_nowait()
                                except queue.Empty:
                                    pass
                            cola.put_nowait(features)
                        except ValueError:
                            pass  # línea malformada — ignorar

            except serial.SerialException:
                print("[SerialBridge] Error de lectura — puerto desconectado", file=sys.stderr)
                break
            except Exception:
                pass

    # ------------------------------------------------------------------
    def enviar_clase(self, clase: int):
        """Envía la clase predicha al Arduino (C,<clase>\\n)."""
        if not self._activo or not self.ser or not self.ser.is_open:
            return
        with self._lock:
            try:
                self.ser.write(f"C,{int(clase)}\n".encode("utf-8"))
            except serial.SerialException:
                pass

    def enviar_angulo(self, angulo: float):
        """Envía un ángulo directo al Arduino (A,<angulo>\\n)."""
        if not self._activo or not self.ser or not self.ser.is_open:
            return
        with self._lock:
            try:
                self.ser.write(f"A,{angulo:.2f}\n".encode("utf-8"))
            except serial.SerialException:
                pass