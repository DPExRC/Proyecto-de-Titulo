# =============================================================================
# dsp.py — Pipeline DSP en Python: filtrado IIR + ventaneo + features
# =============================================================================
# Módulo compartido entre data/captura.py (entrenamiento) y
# src/serial_bridge.py (inferencia en producción).
#
# CapturadorVentanas mantiene el estado de filtrado y ventaneo por canal
# y emite un vector de 12 features (RMS, MAV, WL, ZCR x 3 canales) cada
# vez que se acumulan N_PASO muestras nuevas sobre una ventana llena de
# N_VENTANA muestras filtradas, siguiendo la cadencia de actualización
# de PASO_MS ms (≈ 50 Hz, definida en config.py).
# =============================================================================

# =============================================================================
# dsp.py — Pipeline DSP en Python: filtrado IIR + ventaneo + features
# =============================================================================

import os
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from emg_arm.config import NOMBRES_CANALES, N_VENTANA, N_PASO
from emg_arm.processing.filter import crear_filtros_por_canal
from emg_arm.processing.features import extraer_vector_features


class CapturadorVentanas:
    """Mantiene el estado de filtrado y ventaneo por canal, y emite un
    vector de 12 features cada vez que se acumulan N_PASO muestras
    nuevas sobre una ventana llena de N_VENTANA muestras filtradas."""

    def __init__(self):
        self.filtros   = crear_filtros_por_canal()  # {nombre: FiltroCanal}
        self.buffers   = {n: deque(maxlen=N_VENTANA) for n in NOMBRES_CANALES}
        self._contador = 0

    def reset(self):
        """Reinicia estado del filtro y buffers — llamar al inicio de
        cada sesión de captura para no arrastrar transitorios."""
        for f in self.filtros.values():
            f.reset()
        for b in self.buffers.values():
            b.clear()
        self._contador = 0

    def procesar_trama(self, valores_crudos: list):
        """Procesa una trama de N_CANALES muestras crudas (una por canal)."""
        for nombre, crudo in zip(NOMBRES_CANALES, valores_crudos):
            filtrada = self.filtros[nombre].procesar(crudo)
            self.buffers[nombre].append(filtrada)

        self._contador += 1
        ventana_llena = all(
            len(self.buffers[n]) == N_VENTANA for n in NOMBRES_CANALES
        )

        if ventana_llena and self._contador >= N_PASO:
            self._contador = 0
            ventanas = [list(self.buffers[n]) for n in NOMBRES_CANALES]
            return extraer_vector_features(ventanas)

        return None