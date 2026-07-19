# =============================================================================
# predictor.py — Inferencia en tiempo real: RF regresor multi-salida
# =============================================================================
# Flujo por ciclo (cada PASO_MS, ver config.py):
#   1. Recibe el vector de 12 features (RMS, MAV, WL, ZCR x 3 canales).
#   2. El regresor predice angulo_codo y angulo_muneca simultáneamente.
#   3. Se aplica un Filtro Exponencial (EMA) para suavizar la trayectoria.
#   4. Se aplica una Banda Muerta (Deadband) para eliminar el temblor (jitter).
#   5. Retorna los ángulos filtrados listos para enviar por Serial.
# =============================================================================

import hashlib
import joblib
import os
import sys
import logging

from emg_arm.config import (NOMBRES_FEATURES, COLS_TARGET,
                         ANGULO_MIN, ANGULO_MAX,
                         UMBRAL_BAJO, UMBRAL_ALTO)

logger = logging.getLogger("servos.predictor")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[Predictor] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False

MODEL_DIR     = os.path.join(os.path.dirname(__file__), "..", "..", "models")
PATH_REGRESOR = os.path.join(MODEL_DIR, "modelo_regresor.pkl")

_IDX_RMS_BICEPS    = NOMBRES_FEATURES.index("rms_biceps")
_IDX_RMS_TRICEPS   = NOMBRES_FEATURES.index("rms_triceps")
_IDX_RMS_ANTEBRAZO = NOMBRES_FEATURES.index("rms_antebrazo")


class EMGPredictor:
    """
    Inferencia en tiempo real con regresor multi-salida único, 
    Filtro Suave (EMA) y Banda Muerta (Deadband).
    """

    def __init__(self, alpha_ema=0.2, deadband=2.0):
        self.regresor    = None
        self.regresor_ok = False
        
        # --- Variables de Estado para el Filtro EMA ---
        # alpha_ema: Ponderación del valor nuevo. 
        # (0.2 = 20% predicción nueva, 80% historia). Valores bajos = más suave.
        self.alpha_ema = alpha_ema
        self.ema_codo = ANGULO_MIN
        self.ema_muneca = ANGULO_MIN

        # --- Variables de Estado para Deadband ---
        # deadband: Cambio mínimo en grados para enviar una actualización al servo
        self.deadband = deadband
        self.ultimo_enviado_codo = ANGULO_MIN
        self.ultimo_enviado_muneca = ANGULO_MIN

        self._cargar()

    # ------------------------------------------------------------------
    def _cargar(self):
        p = os.path.abspath(PATH_REGRESOR)
        if not os.path.exists(p):
            logger.warning("Regresor no encontrado: %s", p)
            logger.warning("Ejecuta training/train.py para generarlo. "
                           "Operando con fallback proporcional mientras tanto.")
            return
        try:
            # Verificar que el archivo .pkl es un modelo scikit-learn válido
            # antes de deserializarlo completamente. Esto reduce el vector de
            # ataque de joblib (SEC-01): si el archivo no tiene la estructura
            # esperada de Pipeline/RandomForest, se rechaza antes de ejecutar
            # código arbitrario.
            with open(p, "rb") as f:
                header = f.read(16)
            if not header.startswith(b"\x80"):
                logger.error("El archivo %s no parece ser un pickle válido. Rechazado por seguridad.", p)
                return

            self.regresor = joblib.load(p)
            if hasattr(self.regresor, "named_steps") and "reg" in self.regresor.named_steps:
                self.regresor.named_steps["reg"].n_jobs = 1
            self.regresor_ok = True
            logger.info("Regresor cargado: %s", p)
        except (ValueError, ModuleNotFoundError, KeyError, TypeError, AttributeError) as e:
            logger.error("Error al cargar regresor desde %s: %s", p, e)
        except Exception as e:
            logger.critical("Error inesperado al cargar regresor: %s", e)

    # ------------------------------------------------------------------
    @staticmethod
    def _interpolar(pct_mvc: float) -> float:
        if pct_mvc < UMBRAL_BAJO:
            return ANGULO_MIN
        if pct_mvc >= UMBRAL_ALTO:
            return ANGULO_MAX
        proporcion = (pct_mvc - UMBRAL_BAJO) / (UMBRAL_ALTO - UMBRAL_BAJO)
        return ANGULO_MIN + proporcion * (ANGULO_MAX - ANGULO_MIN)

    def _fallback_codo(self, features: list) -> float:
        pct_biceps  = features[_IDX_RMS_BICEPS]
        pct_triceps = features[_IDX_RMS_TRICEPS]
        neto = max(pct_biceps - pct_triceps, 0.0)
        return self._interpolar(neto)

    def _fallback_muneca(self, features: list) -> float:
        pct_antebrazo = features[_IDX_RMS_ANTEBRAZO]
        return self._interpolar(pct_antebrazo)

    def _fallback(self, features: list) -> dict:
        return {
            "angulo_codo":   self._fallback_codo(features),
            "angulo_muneca": self._fallback_muneca(features),
        }

    # ------------------------------------------------------------------
    def predecir_angulos(self, features: list) -> dict:
        
        # 1. Obtener predicción cruda (Regresor o Fallback)
        raw_codo = ANGULO_MIN
        raw_muneca = ANGULO_MIN
        usar_fallback = True

        if len(features) == len(NOMBRES_FEATURES) and self.regresor_ok:
            try:
                pred = self.regresor.predict([features])[0]
                raw_codo   = float(min(max(pred[0], ANGULO_MIN), ANGULO_MAX))
                raw_muneca = float(min(max(pred[1], ANGULO_MIN), ANGULO_MAX))
                usar_fallback = False
            except Exception as e:
                logger.error("Error en inferencia del regresor: %s. Usando fallback.", e)

        if usar_fallback:
            if len(features) >= 9:
                fallback_res = self._fallback(features)
                raw_codo, raw_muneca = fallback_res["angulo_codo"], fallback_res["angulo_muneca"]
            else:
                logger.warning("Features incompletas (%d).", len(features))

        # 2. Aplicar Filtro Suave (Promedio Móvil Exponencial - EMA)
        self.ema_codo = (self.alpha_ema * raw_codo) + ((1.0 - self.alpha_ema) * self.ema_codo)
        self.ema_muneca = (self.alpha_ema * raw_muneca) + ((1.0 - self.alpha_ema) * self.ema_muneca)

        # 3. Aplicar Banda Muerta (Deadband)
        # Solo actualiza el ángulo final si la diferencia supera la banda muerta
        if abs(self.ema_codo - self.ultimo_enviado_codo) >= self.deadband:
            self.ultimo_enviado_codo = self.ema_codo

        if abs(self.ema_muneca - self.ultimo_enviado_muneca) >= self.deadband:
            self.ultimo_enviado_muneca = self.ema_muneca

        # 4. Retornar ángulos redondeados listos para Arduino
        return {
            "angulo_codo": round(self.ultimo_enviado_codo, 1),
            "angulo_muneca": round(self.ultimo_enviado_muneca, 1)
        }
