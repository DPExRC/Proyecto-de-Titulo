# =============================================================================
# config.py — Parámetros globales del sistema EMG
# =============================================================================

import os

# ---------------------------------------------------------------------------
# Puerto serial
# ---------------------------------------------------------------------------
PORT     = "COM3"       # Cambiar según sistema operativo (Linux: /dev/ttyUSB0)
BAUDRATE = 115200


# ---------------------------------------------------------------------------
# Señal EMG
# ---------------------------------------------------------------------------
N_CANALES       = 2
FS              = 500.0    # Hz por canal (muestreo alternado, 1000 Hz total)
VENTANA_MS      = 250      # ms
PASO_MS         = 20       # ms
N_VENTANA       = int(FS * VENTANA_MS / 1000)  # 125 muestras por canal
N_PASO          = int(FS * PASO_MS    / 1000)  #   10 muestras por canal

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
# Vector: [rms_ch0(%MVC), zcr_ch0, rms_ch1(%MVC), zcr_ch1]
N_FEATURES = 4
NOMBRES_FEATURES = ["rms_biceps", "zcr_biceps", "rms_triceps", "zcr_triceps"]

# ---------------------------------------------------------------------------
# Clases
# ---------------------------------------------------------------------------
CLASES       = {0: "REPOSO", 1: "FLEXION", 2: "EXTENSION"}
N_CLASES     = 3

# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "modelo_emg.pkl")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data",   "datos_emg.csv")

# ---------------------------------------------------------------------------
# Control servo
# ---------------------------------------------------------------------------
ANGULO_MIN = 0.0
ANGULO_MAX = 180.0
ANGULO_REPOSO    = 90.0
ANGULO_FLEXION   = 170.0
ANGULO_EXTENSION = 10.0

UMBRAL_REPOSO_PCT = 20.0
UMBRAL_ACTIVO_PCT = 80.0

INTERVALO_CONTROL = PASO_MS / 1000.0   # s — misma cadencia que el firmware

# ---------------------------------------------------------------------------
# Captura de dataset
# ---------------------------------------------------------------------------
DURACION_CAPTURA_S = 5      # segundos de captura por etiqueta
MIN_REGISTROS_CLASE = 30    # mínimo recomendado por clase para entrenamiento