# =============================================================================
# verification.py — Verificación de software del pipeline, sin hardware
# =============================================================================
# Cada función valida UNA cosa específica y retorna las métricas crudas
# (no un bool ya decidido) — las decisiones de aceptación (umbrales) viven
# en los tests, no aquí, para que el reporte sirva también para el
# informe técnico sin tener que re-derivar los números.
#
# Principio de diseño: cada verificación debe poder FALLAR de verdad si
# el pipeline se rompe. Se evitan deliberadamente:
#   - Métricas hardcodeadas/simuladas (ej. "latencia = 12.0" fijo).
#   - Comparar la salida de una función contra sí misma con la misma
#     fórmula (ej. RMS de la señal ya filtrada contra RMS de la señal
#     ya filtrada) — eso no prueba que el filtro haga nada.
#   - Rangos que el propio código de producción ya garantiza por
#     construcción (ej. un ángulo que el predictor ya clampea con
#     min/max no puede "fallar" el rango [0,180] nunca).
# =============================================================================

import os
import sys
import time
import json
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.config import FS, N_VENTANA, N_FEATURES_POR_CANAL, NOMBRES_FEATURES
from src.processing.features import extraer_vector_features
from src.processing.filter import FiltroEMG
from src.processing.calibration import RUTA_CALIBRACION_DEFAULT
from src.core.serial_bridge import parsear_trama_emg
from src.models.predictor import EMGPredictor


# =============================================================================
# 1. Parseo serial
# =============================================================================
def verificar_parseo_serial() -> dict:
    """Valida el parseo tanto en el camino feliz como en casos que deben
    ser rechazados — una sola aserción de 'parseo_ok' no distingue si el
    parser es correcto o simplemente permisivo con todo."""
    valida = parsear_trama_emg("S,512,480,300")
    rechaza_prefijo_invalido = parsear_trama_emg("X,512,480,300")
    rechaza_canales_faltantes = parsear_trama_emg("S,512,480")
    rechaza_valor_no_numerico = parsear_trama_emg("S,512,abc,300")

    return {
        "trama_valida": valida,
        "rechaza_prefijo_invalido": rechaza_prefijo_invalido is None,
        "rechaza_canales_faltantes": rechaza_canales_faltantes is None,
        "rechaza_valor_no_numerico": rechaza_valor_no_numerico is None,
    }


# =============================================================================
# 2. Filtro — discrimina banda pasante vs. fuera de banda
# =============================================================================
def verificar_filtro_discrimina_bandas(fs: float = FS) -> dict:
    """Compara la salida del filtro contra la ENTRADA cruda (no contra sí
    misma), con dos señales sintéticas: una dentro de la banda pasante
    (debe pasar con poca atenuación) y otra fuera de banda (debe
    atenuarse fuertemente). Esto sí puede detectar un filtro roto,
    invertido, o que actúa como passthrough."""
    duracion_s = 1.0
    t = np.arange(0, duracion_s, 1.0 / fs)
    amplitud = 100.0

    f_dentro = 60.0   # Hz, dentro de la banda útil sEMG (20-150 Hz)
    f_fuera = 5.0      # Hz, por debajo del corte inferior (20 Hz)

    señal_dentro = amplitud * np.sin(2 * np.pi * f_dentro * t)
    señal_fuera = amplitud * np.sin(2 * np.pi * f_fuera * t)

    filtro_a = FiltroEMG(fs=fs)
    filtro_b = FiltroEMG(fs=fs)

    salida_dentro = np.array([filtro_a.procesar(float(x)) for x in señal_dentro])
    salida_fuera = np.array([filtro_b.procesar(float(x)) for x in señal_fuera])

    # Ignorar el transitorio inicial del filtro al medir amplitud
    n_ignorar = len(t) // 3
    rms_entrada = amplitud / np.sqrt(2)
    rms_salida_dentro = float(np.sqrt(np.mean(salida_dentro[n_ignorar:] ** 2)))
    rms_salida_fuera = float(np.sqrt(np.mean(salida_fuera[n_ignorar:] ** 2)))

    atenuacion_dentro_db = 20 * np.log10(max(rms_salida_dentro, 1e-9) / rms_entrada)
    atenuacion_fuera_db = 20 * np.log10(max(rms_salida_fuera, 1e-9) / rms_entrada)

    return {
        "rms_entrada": rms_entrada,
        "rms_salida_dentro_banda": rms_salida_dentro,
        "rms_salida_fuera_banda": rms_salida_fuera,
        "atenuacion_dentro_db": float(atenuacion_dentro_db),
        "atenuacion_fuera_db": float(atenuacion_fuera_db),
    }


# =============================================================================
# 3. Features — validez estructural y por rango físico
# =============================================================================
def verificar_features_validas() -> dict:
    """Verifica límites físicos reales de cada tipo de feature, no solo
    'son números finitos'. RMS/MAV/WL no pueden ser negativos; ZC no
    puede exceder el número de muestras de la ventana."""
    señal = np.sin(np.linspace(0, 4 * np.pi, N_VENTANA))
    features = extraer_vector_features([señal, señal, señal])

    idx_rms = [i for i, n in enumerate(NOMBRES_FEATURES) if n.startswith("rms_")]
    idx_mav = [i for i, n in enumerate(NOMBRES_FEATURES) if n.startswith("mav_")]
    idx_wl  = [i for i, n in enumerate(NOMBRES_FEATURES) if n.startswith("wl_")]
    idx_zc  = [i for i, n in enumerate(NOMBRES_FEATURES) if n.startswith("zc_")]

    return {
        "longitud": len(features),
        "todos_finitos": bool(np.all(np.isfinite(features))),
        "rms_no_negativo": bool(all(features[i] >= 0 for i in idx_rms)),
        "mav_no_negativo": bool(all(features[i] >= 0 for i in idx_mav)),
        "wl_no_negativo": bool(all(features[i] >= 0 for i in idx_wl)),
        "zc_en_rango_fisico": bool(all(0 <= features[i] <= N_VENTANA for i in idx_zc)),
    }


# =============================================================================
# 4. Predictor — responde de forma coherente a la activación muscular
# =============================================================================
def verificar_predictor_responde_a_activacion() -> dict:
    """Compara la predicción en reposo (todos los canales en 0% MVC)
    contra la predicción con el bíceps activado — el codo debe FLEXIONAR
    (aumentar de ángulo), no simplemente 'estar entre 0 y 180', que el
    código ya garantiza por construcción sin importar si el modelo
    funciona.

    alpha_ema=1.0 y deadband=0.0 desactivan el suavizado temporal para
    medir la respuesta inmediata a cada llamada, sin arrastrar estado
    de llamadas anteriores."""
    idx_biceps = NOMBRES_FEATURES.index("rms_biceps")
    n_feat = len(NOMBRES_FEATURES)

    predictor = EMGPredictor(alpha_ema=1.0, deadband=0.0)

    features_reposo = [0.0] * n_feat
    features_biceps_activo = [0.0] * n_feat
    features_biceps_activo[idx_biceps] = 95.0  # % MVC, activación fuerte

    codo_reposo = predictor.predecir_angulos(features_reposo)["angulo_codo"]
    codo_activo = predictor.predecir_angulos(features_biceps_activo)["angulo_codo"]

    return {
        "modelo_cargado": predictor.regresor_ok,
        "codo_reposo": codo_reposo,
        "codo_activo": codo_activo,
        "diferencia": codo_activo - codo_reposo,
    }


# =============================================================================
# 5. Latencia — medida real, no simulada
# =============================================================================
def medir_latencia_pipeline(n_repeticiones: int = 50) -> dict:
    """Mide el tiempo real de un ciclo completo (filtrado + features +
    inferencia) con time.perf_counter(), repitiendo n_repeticiones veces
    para reportar promedio y máximo. Si el pipeline se vuelve más lento
    en el futuro, este número cambia — a diferencia de un valor fijo."""
    filtro = FiltroEMG(fs=FS)
    predictor = EMGPredictor()
    señal = np.sin(np.linspace(0, 4 * np.pi, N_VENTANA))

    tiempos_ms = []
    for _ in range(n_repeticiones):
        t0 = time.perf_counter()

        filtrada = [filtro.procesar(float(v)) for v in señal]
        features = extraer_vector_features([filtrada, filtrada, filtrada])
        predictor.predecir_angulos(features)

        tiempos_ms.append((time.perf_counter() - t0) * 1000.0)

    return {
        "promedio_ms": float(np.mean(tiempos_ms)),
        "maximo_ms": float(np.max(tiempos_ms)),
        "n_repeticiones": n_repeticiones,
    }


# =============================================================================
# 6. SNR — relación señal/ruido por canal, desde la calibración ya capturada
# =============================================================================
def calcular_snr(ruta_calibracion: str = RUTA_CALIBRACION_DEFAULT) -> dict:
    """Calcula SNR_dB = 20*log10(RMS_MVC / RMS_baseline) por canal, usando
    los valores de RMS ya capturados en el protocolo de calibración
    (reposo + MVC) — no requiere instrumentación ni captura adicional.

    Es una aproximación: usa el RMS de reposo como proxy del piso de
    ruido, no una medición de ruido aislada con electrodos en corto. Si
    se necesita el SNR "de manual" (ruido puro, sin señal fisiológica de
    fondo), esta función no lo reemplaza — solo aprovecha datos que ya
    existen para dar un número trazable en vez de la afirmación sin
    respaldo que hay hoy en el Cap. 7."""
    with open(ruta_calibracion, "r", encoding="utf-8") as f:
        datos = json.load(f)

    features = datos["features"]
    baseline = datos["baseline"]
    mvc = datos["mvc"]

    snr_por_canal = {}
    for canal in ("biceps", "triceps", "antebrazo"):
        idx_rms = features.index(f"rms_{canal}")
        rms_base = baseline[idx_rms]
        rms_mvc = mvc[idx_rms]
        snr_por_canal[canal] = (
            float(20.0 * np.log10(rms_mvc / rms_base)) if rms_base > 0 else None
        )

    validos = [v for v in snr_por_canal.values() if v is not None]
    return {
        "snr_db_por_canal": snr_por_canal,
        "snr_db_promedio": float(np.mean(validos)) if validos else None,
        "fuente": os.path.abspath(ruta_calibracion),
    }


# =============================================================================
# Reporte combinado — para el informe técnico, no para basar tests en él
# =============================================================================
def generar_reporte_completo() -> dict:
    """Junta todas las verificaciones en un solo dict, útil para volcar
    a un archivo/tabla en el informe. Los TESTS deben llamar a las
    funciones individuales de arriba, no este reporte combinado — así
    un fallo señala exactamente cuál verificación se rompió."""
    return {
        "parseo_serial": verificar_parseo_serial(),
        "filtro": verificar_filtro_discrimina_bandas(),
        "features": verificar_features_validas(),
        "predictor": verificar_predictor_responde_a_activacion(),
        "latencia": medir_latencia_pipeline(),
        "snr": calcular_snr(),
    }


if __name__ == "__main__":
    import json
    print(json.dumps(generar_reporte_completo(), indent=2))