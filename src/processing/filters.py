# -*- coding: utf-8 -*-
"""filters.py: Filtrado de señales EMG."""
from scipy.signal import butter, sosfilt
import src.config as cfg


class EMGFilter:
    """
    Butterworth pasa-banda implementado con sosfilt (second-order sections).

    Por qué sosfilt en lugar de lfilter:
      - lfilter usa coeficientes b/a que con orden >= 4 acumulan
        errores numéricos de punto flotante.
      - sosfilt encadena secciones de orden 2, numéricamente estable
        y sin distorsión de fase adicional en tiempo real.
    """

    def __init__(self):
        self.sos = self._butter_bandpass(
            cfg.LOWCUT, cfg.HIGHCUT, cfg.FS, order=cfg.FILTRO_ORDEN
        )

    @staticmethod
    def _butter_bandpass(lowcut, highcut, fs, order=4):
        nyq  = 0.5 * fs
        low  = lowcut  / nyq
        high = highcut / nyq
        return butter(order, [low, high], btype="band", output="sos")

    def aplicar_filtro(self, datos_crudos):
        """Aplica el filtro Butterworth pasa-banda. Entrada/salida: np.ndarray."""
        return sosfilt(self.sos, datos_crudos)