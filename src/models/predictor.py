# =============================================================================
# predictor.py — Inferencia en tiempo real: RF regresor multi-salida
# =============================================================================
# Flujo por ciclo (cada PASO_MS, ver config.py):
#   1. Recibe el vector de 12 features (RMS, MAV, WL, ZCR x 3 canales),
#      ya calculado aguas arriba (módulo de DSP en Python).
#   2. El regresor predice angulo_codo y angulo_muneca simultáneamente.
#   3. Si el modelo no está disponible, usa un fallback proporcional por
#      umbral (idéntico en espíritu al firmware Arduino original), no una
#      clasificación discreta — no existe etapa de clasificación en este
#      pipeline.
#
# Convención de ángulo (fijada para todo el proyecto): reposo = 0° en
# ambos DOF.
#   - Codo: bidireccional. Bíceps incrementa el ángulo (flexión).
#     Tríceps acelera el retorno hacia 0°: nunca produce ángulos
#     negativos, el piso del rango es 0°.
#   - Muñeca: unidireccional. Pronator teres (antebrazo) incrementa el ángulo
#     hacia 180°; en ausencia de activación, el ángulo decae hacia 0°
#     (el decaimiento en sí lo gestiona el filtro exponencial/slew-rate
#     aguas abajo de este módulo, no es responsabilidad del predictor).
#
# IMPORTANTE — normalización %MVC pendiente:
#   Este módulo asume que los valores rms_* del vector de entrada ya
#   vienen normalizados a %MVC (0-100, calibrados por sesión), igual
#   que en el firmware Arduino original. Actualmente no existe en el
#   proyecto un módulo Python de calibración/normalización (baseline +
#   MVC por canal) — debe implementarse antes de que este predictor
#   pueda usarse con datos reales. Si features.py entrega RMS crudo
#   (no normalizado), tanto UMBRAL_BAJO/UMBRAL_ALTO como el fallback de
#   este archivo van a operar sobre una escala incorrecta.
# =============================================================================

import joblib
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import (NOMBRES_FEATURES, COLS_TARGET,
                         ANGULO_MIN, ANGULO_MAX,
                         UMBRAL_BAJO, UMBRAL_ALTO)

MODEL_DIR     = os.path.join(os.path.dirname(__file__), "..", "..", "models")
PATH_REGRESOR = os.path.join(MODEL_DIR, "modelo_regresor.pkl")

# Índices de las features relevantes para el fallback, resueltos por
# nombre en vez de hardcodear posiciones — evita romperse si el orden
# de NOMBRES_FEATURES cambia en config.py.
_IDX_RMS_BICEPS    = NOMBRES_FEATURES.index("rms_biceps")
_IDX_RMS_TRICEPS   = NOMBRES_FEATURES.index("rms_triceps")
_IDX_RMS_ANTEBRAZO = NOMBRES_FEATURES.index("rms_antebrazo")


class EMGPredictor:
    """
    Inferencia en tiempo real con regresor multi-salida único.

    predecir_angulos(features) → {"angulo_codo": float, "angulo_muneca": float}
    """

    def __init__(self):
        self.regresor    = None
        self.regresor_ok = False
        self._cargar()

    # ------------------------------------------------------------------
    def _cargar(self):
        p = os.path.abspath(PATH_REGRESOR)
        if not os.path.exists(p):
            print(f"[Predictor] Regresor no encontrado: {p}")
            print("[Predictor] Ejecuta training/train.py para generarlo. "
                  "Operando con fallback proporcional mientras tanto.")
            return
        try:
            self.regresor = joblib.load(p)
            self.regresor_ok = True
            print(f"[Predictor] Regresor cargado: {p}")
        except Exception as e:
            print(f"[Predictor] Error al cargar regresor: {e}")

    # ------------------------------------------------------------------
    @staticmethod
    def _interpolar(pct_mvc: float) -> float:
        """Mapeo lineal de %MVC normalizado a ángulo [0, 180], saturado
        fuera de [UMBRAL_BAJO, UMBRAL_ALTO]. Misma lógica que el mapeo
        RF-08 del firmware Arduino, aplicada aquí como fallback."""
        if pct_mvc < UMBRAL_BAJO:
            return ANGULO_MIN
        if pct_mvc >= UMBRAL_ALTO:
            return ANGULO_MAX
        proporcion = (pct_mvc - UMBRAL_BAJO) / (UMBRAL_ALTO - UMBRAL_BAJO)
        return ANGULO_MIN + proporcion * (ANGULO_MAX - ANGULO_MIN)

    def _fallback_codo(self, features: list) -> float:
        """Codo bidireccional: bíceps empuja hacia 180°, tríceps acelera
        el retorno hacia 0° (sin producir ángulos negativos)."""
        pct_biceps  = features[_IDX_RMS_BICEPS]
        pct_triceps = features[_IDX_RMS_TRICEPS]
        neto = max(pct_biceps - pct_triceps, 0.0)
        return self._interpolar(neto)

    def _fallback_muneca(self, features: list) -> float:
        """Muñeca unidireccional: pronator teres (antebrazo) empuja hacia 180°."""
        pct_antebrazo = features[_IDX_RMS_ANTEBRAZO]
        return self._interpolar(pct_antebrazo)

    def _fallback(self, features: list) -> dict:
        return {
            "angulo_codo":   self._fallback_codo(features),
            "angulo_muneca": self._fallback_muneca(features),
        }

    # ------------------------------------------------------------------
    def predecir_angulos(self, features: list) -> dict:
        """
        Parámetros
        ----------
        features : vector de 12 valores, en el orden de
                   config.NOMBRES_FEATURES (RMS, MAV, WL, ZCR x 3 canales).

        Retorna
        -------
        dict con claves "angulo_codo" y "angulo_muneca", cada uno en
        [ANGULO_MIN, ANGULO_MAX].
        """
        if len(features) != len(NOMBRES_FEATURES):
            print(f"[Predictor] ADVERTENCIA: se esperaban "
                  f"{len(NOMBRES_FEATURES)} features, llegaron "
                  f"{len(features)}. Usando fallback.")
            return self._fallback(features) if len(features) >= 9 else \
                {"angulo_codo": ANGULO_MIN, "angulo_muneca": ANGULO_MIN}

        if self.regresor_ok:
            try:
                pred = self.regresor.predict([features])[0]  # [codo, muñeca]
                angulo_codo   = float(min(max(pred[0], ANGULO_MIN), ANGULO_MAX))
                angulo_muneca = float(min(max(pred[1], ANGULO_MIN), ANGULO_MAX))
                return {"angulo_codo": angulo_codo, "angulo_muneca": angulo_muneca}
            except Exception as e:
                print(f"[Predictor] Error en inferencia del regresor: {e}. "
                      f"Usando fallback.")

        return self._fallback(features)