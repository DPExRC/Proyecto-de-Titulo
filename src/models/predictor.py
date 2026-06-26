# =============================================================================
# predictor.py — Pipeline dual: clasificador + regresor EMG
# =============================================================================
# Flujo por ciclo (cada 20 ms):
#   1. Clasificador RF  → determina qué músculo manda (REPOSO/FLEXION/EXTENSION)
#   2. Regresor RF      → predice el ángulo continuo a partir de las 4 features
#   3. Si clase == REPOSO → ángulo forzado a 180° (zona muerta)
#      Si clase == FLEXION/EXTENSION → se usa el ángulo del regresor
#
# Ambos modelos se cargan desde models/modelo_clasificador.pkl
#                                y models/modelo_regresor.pkl
# =============================================================================

import joblib
import os
import sys

## Falta configurar los import de src.config, no existen en el archivo original y algunos como umbral_reposo, son el baselina y mvc
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.config import (CLASES, ANGULO_MIN, ANGULO_MAX,
                         UMBRAL_REPOSO_PCT, UMBRAL_ACTIVO_PCT)

MODEL_DIR     = os.path.join(os.path.dirname(__file__), "..", "..", "models")
PATH_CLASIF   = os.path.join(MODEL_DIR, "modelo_clasificador.pkl")
PATH_REGRESOR = os.path.join(MODEL_DIR, "modelo_regresor.pkl")


class EMGPredictor:
    """
    Inferencia en tiempo real con pipeline dual.

    predecir_angulo(features) → (clase: int, angulo: float)
      - clase:  0=REPOSO, 1=FLEXION, 2=EXTENSION
      - angulo: valor continuo en [0.0, 180.0] actualizado cada 20 ms
    """

    def __init__(self):
        self.clasif   = None
        self.regresor = None
        self.clasif_ok   = False
        self.regresor_ok = False
        self._cargar()

    # ------------------------------------------------------------------
    def _cargar(self):
        for path, attr_ok, attr_mod, nombre in [
            (PATH_CLASIF,   "clasif_ok",   "clasif",   "Clasificador"),
            (PATH_REGRESOR, "regresor_ok", "regresor", "Regresor")
        ]:
            p = os.path.abspath(path)
            if not os.path.exists(p):
                print(f"[Predictor] {nombre} no encontrado: {p}")
                print("[Predictor] Ejecuta training/train.py para generar los modelos.")
                continue
            try:
                setattr(self, attr_mod, joblib.load(p))
                setattr(self, attr_ok,  True)
                print(f"[Predictor] {nombre} cargado: {p}")
            except Exception as e:
                print(f"[Predictor] Error al cargar {nombre}: {e}")

    # ------------------------------------------------------------------
    def _fallback_clase(self, features: list) -> int:
        """Clasificación por umbral cuando el modelo no está disponible."""
        rms0, _, rms1, _ = features
        if rms0 > UMBRAL_ACTIVO_PCT and rms1 < UMBRAL_REPOSO_PCT:
            return 1  # FLEXION
        elif rms1 > UMBRAL_ACTIVO_PCT and rms0 < UMBRAL_REPOSO_PCT:
            return 2  # EXTENSION
        return 0      # REPOSO

    def _fallback_angulo(self, features: list, clase: int) -> float:
        """Mapeo lineal de %MVC cuando el regresor no está disponible."""
        rms0 = features[0]   # %MVC bíceps
        rms1 = features[2]   # %MVC tríceps
        if clase == 1:
            proporcion = max(rms0 - UMBRAL_ACTIVO_PCT, 0.0) / (100.0 - UMBRAL_ACTIVO_PCT)
            return ANGULO_MAX - proporcion * (ANGULO_MAX - ANGULO_MIN)
        elif clase == 2:
            proporcion = max(rms1 - UMBRAL_ACTIVO_PCT, 0.0) / (100.0 - UMBRAL_ACTIVO_PCT)
            return ANGULO_MIN + proporcion * (ANGULO_MAX - ANGULO_MIN)
        return ANGULO_MAX   # REPOSO

    # ------------------------------------------------------------------
    def predecir_angulo(self, features: list) -> tuple:
        """
        Parámetros
        ----------
        features : [rms_biceps_%mvc, zcr_biceps, rms_triceps_%mvc, zcr_triceps]

        Retorna
        -------
        (clase: int, angulo: float)
        """
        # --- Clasificador ---
        if self.clasif_ok:
            try:
                clase = int(self.clasif.predict([features])[0])
                clase = max(0, min(2, clase))
            except Exception as e:
                print(f"[Predictor] Error clasificador: {e}")
                clase = self._fallback_clase(features)
        else:
            clase = self._fallback_clase(features)

        # --- Zona muerta: REPOSO fuerza 180° sin consultar el regresor ---
        if clase == 0:
            return 0, ANGULO_MAX

        # --- Regresor ---
        if self.regresor_ok:
            try:
                angulo = float(self.regresor.predict([features])[0])
                angulo = max(ANGULO_MIN, min(ANGULO_MAX, angulo))
            except Exception as e:
                print(f"[Predictor] Error regresor: {e}")
                angulo = self._fallback_angulo(features, clase)
        else:
            angulo = self._fallback_angulo(features, clase)

        return clase, angulo

    def nombre_clase(self, clase: int) -> str:
        return CLASES.get(clase, "DESCONOCIDO")