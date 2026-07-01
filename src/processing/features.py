# =============================================================================
# features.py — Extracción de características temporales por canal
# =============================================================================
# Módulo nuevo. Centraliza el cálculo de RMS, MAV, WL y ZC para que
# captura.py (entrenamiento) y predictor.py (inferencia en tiempo real)
# usen exactamente la misma implementación. No reimplementar estas
# fórmulas por separado en otros scripts — importar desde aquí.
#
# Asume que cada canal llega como un arreglo 1D de muestras filtradas
# (ya pasadas por el filtro IIR Butterworth), de longitud N_VENTANA
# definida en config.py.
# =============================================================================

import numpy as np


def rms(ventana: np.ndarray) -> float:
    """Valor cuadrático medio (Root Mean Square)."""
    return float(np.sqrt(np.mean(np.square(ventana))))


def mav(ventana: np.ndarray) -> float:
    """Valor absoluto medio (Mean Absolute Value)."""
    return float(np.mean(np.abs(ventana)))


def wl(ventana: np.ndarray) -> float:
    """Longitud de forma de onda (Waveform Length):
    suma de las diferencias absolutas entre muestras consecutivas."""
    return float(np.sum(np.abs(np.diff(ventana))))


def zc(ventana: np.ndarray, umbral: float = 0.0) -> int:
    """Cruces por cero (Zero Crossings), con umbral mínimo de variación
    para descartar cruces producidos por ruido de baja amplitud."""
    signos = np.sign(ventana)
    # Reemplaza ceros exactos por el signo de la muestra previa para no
    # contar falsos cruces en tramos planos.
    for i in range(1, len(signos)):
        if signos[i] == 0:
            signos[i] = signos[i - 1]

    cambios = np.diff(signos) != 0
    if umbral > 0.0:
        amplitud_suficiente = np.abs(np.diff(ventana)) >= umbral
        cambios = cambios & amplitud_suficiente
    return int(np.sum(cambios))


def extraer_features_canal(ventana: np.ndarray, zc_umbral: float = 0.0) -> list:
    """Retorna [rms, mav, wl, zc] para un canal, en el orden fijado por
    NOMBRES_FEATURES_POR_CANAL en config.py. No alterar el orden sin
    actualizar config.py también."""
    return [
        rms(ventana),
        mav(ventana),
        wl(ventana),
        zc(ventana, umbral=zc_umbral),
    ]


def extraer_vector_features(ventanas_por_canal: list, zc_umbral: float = 0.0) -> list:
    """Recibe una lista de arreglos (uno por canal, en el orden de
    NOMBRES_CANALES de config.py) y retorna el vector concatenado de
    12 features, listo para alimentar al regresor.

    ventanas_por_canal: [ventana_biceps, ventana_triceps, ventana_antebrazo]
    """
    vector = []
    for ventana in ventanas_por_canal:
        vector.extend(extraer_features_canal(ventana, zc_umbral=zc_umbral))
    return vector


if __name__ == "__main__":
    # Prueba mínima de cordura: señal sinusoidal sintética.
    t = np.linspace(0, 1, 250, endpoint=False)
    senal = 50.0 * np.sin(2 * np.pi * 5 * t)
    print("RMS esperado ≈ A/√2 = 35.36 — calculado:", rms(senal))
    print("MAV:", mav(senal))
    print("WL:", wl(senal))
    print("ZC:", zc(senal))