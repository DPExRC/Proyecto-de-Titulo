# -*- coding: utf-8 -*-
"""features.py: Extracción de características de ventanas temporales."""
import numpy as np

class FeatureExtractor:
    @staticmethod
    def calcular_rms(senal_ventana):
        """Calcula el Root Mean Square (RMS)."""
        if len(senal_ventana) == 0:
            return 0.0
        return np.sqrt(np.mean(senal_ventana ** 2))

    @staticmethod
    def calcular_mav(senal_ventana):
        """Calcula el Mean Absolute Value (MAV) por si decides usarlo en el Random Forest."""
        return np.mean(np.abs(senal_ventana))