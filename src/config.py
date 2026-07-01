# =============================================================================
# config.py — Parámetros globales del sistema EMG (v3.0)
# =============================================================================
# Arquitectura confirmada:
#   - 3 canales sEMG: bíceps braquial, tríceps braquial, pronator teres (antebrazo)
#   - 2 DOF controlados por EMG:
#       DOF 1 — Codo: bidireccional, par antagonista biceps/triceps.
#               Reposo = 0°. Activación de bíceps incrementa el ángulo
#               (flexión). Activación de tríceps acelera el retorno hacia 0°
#               (no produce ángulos negativos; el piso del rango es 0°).
#       DOF 2 — Muñeca: unidireccional, canal único pronator teres (antebrazo).
#               Reposo = 0°. Activación incrementa el ángulo hacia 180°.
#   - Vector de características: RMS, MAV, WL, ZC por canal (4 x 3 = 12).
#   - Modelo: un único RandomForestRegressor multi-salida
#     (angulo_codo, angulo_muneca). Sin etapa de clasificación — el
#     gating de reposo/ruido se resuelve mediante UMBRAL_BAJO/UMBRAL_ALTO
#     (%MVC), el filtro exponencial asimétrico y el limitador de
#     slew-rate, igual que en el firmware Arduino.
#
# ── PENDIENTE DE CONFIRMACIÓN ────────────────────────────────────────────
#   UMBRAL_BAJO/UMBRAL_ALTO se fijaron en 12.0/90.0 para que coincidan con
#   el firmware Arduino (emg_v3.ino), que es el único componente del
#   proyecto con evidencia de validación sobre hardware real. La versión
#   anterior de este archivo usaba 20.0/80.0 (UMBRAL_REPOSO_PCT /
#   UMBRAL_ACTIVO_PCT) — verificar cuál par fue efectivamente validado
#   experimentalmente antes de la entrega final, y unificar en ambos
#   lugares (firmware y Python) si se confirma un valor distinto.
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
N_CANALES   = 3                              # biceps, triceps, antebrazo
FS_TOTAL    = 1000.0                         # Hz, tasa total del ADC (Arduino)
FS          = FS_TOTAL / N_CANALES           # Hz efectivos por canal ≈ 333.33
VENTANA_MS  = 250                            # ms
PASO_MS     = 20                             # ms
N_VENTANA   = round(FS * VENTANA_MS / 1000)  # ≈ 83 muestras por canal
N_PASO      = round(FS * PASO_MS / 1000)     # ≈ 7 muestras por canal

# Nyquist efectivo por canal, derivado de FS (ver Capítulo 1, sección 1.4.1)
NYQUIST_EFECTIVO_HZ = FS / 2.0               # ≈ 166.7 Hz
FILTRO_CORTE_HZ     = 150.0                  # margen de seguridad bajo Nyquist

# ---------------------------------------------------------------------------
# Canales — nombres explícitos por índice, evita errores de orden
# ---------------------------------------------------------------------------
NOMBRES_CANALES = ["biceps", "triceps", "antebrazo"]

# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------
# Vector: 4 características temporales (RMS, MAV, WL, ZC) por cada uno de
# los 3 canales = 12 columnas. Orden fijo, usado tanto en captura como en
# entrenamiento e inferencia — no reordenar sin actualizar los tres scripts.
NOMBRES_FEATURES_POR_CANAL = ["rms", "mav", "wl", "zc"]
N_FEATURES_POR_CANAL = len(NOMBRES_FEATURES_POR_CANAL)
N_FEATURES = N_FEATURES_POR_CANAL * N_CANALES  # 12

NOMBRES_FEATURES = [
    f"{feat}_{canal}"
    for canal in NOMBRES_CANALES
    for feat in NOMBRES_FEATURES_POR_CANAL
]
# Resultado: ['rms_biceps', 'mav_biceps', 'wl_biceps', 'zc_biceps',
#             'rms_triceps', 'mav_triceps', 'wl_triceps', 'zc_triceps',
#             'rms_antebrazo', 'mav_antebrazo', 'wl_antebrazo', 'zc_antebrazo']

# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "modelo_regresor.pkl")
DATA_PATH  = os.path.join(os.path.dirname(__file__), "..", "data",   "datos_emg.csv")

# ---------------------------------------------------------------------------
# Control servo — 2 DOF, reposo = 0° en ambos (ver nota de cabecera)
# ---------------------------------------------------------------------------
ANGULO_MIN = 0.0
ANGULO_MAX = 180.0

COL_ANGULO_CODO   = "angulo_codo"
COL_ANGULO_MUNECA = "angulo_muneca"
COLS_TARGET       = [COL_ANGULO_CODO, COL_ANGULO_MUNECA]

# Umbrales de gating en %MVC — alineados con el firmware Arduino (ver nota
# de cabecera sobre el valor pendiente de confirmar).
UMBRAL_BAJO  = 12.0   # %MVC bajo el cual se considera reposo (ángulo → 0°)
UMBRAL_ALTO  = 90.0   # %MVC sobre el cual se satura el ángulo (→ 180°)

INTERVALO_CONTROL = PASO_MS / 1000.0   # s — misma cadencia que el firmware

# ---------------------------------------------------------------------------
# Captura de dataset
# ---------------------------------------------------------------------------
DURACION_CAPTURA_S  = 5      # segundos de captura por etiqueta/ángulo objetivo
MIN_REGISTROS_CLASE = 30     # mínimo recomendado por combinación de ángulos