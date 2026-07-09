# =============================================================================
# filtro.py — Procesamiento Digital de Señales (DSP) en tiempo real
# =============================================================================
# Ubicación prevista: src/processing/filtro.py
#
# Aplica secuencialmente, por canal:
#   1. Filtro Notch (50 Hz) — elimina interferencia de la red eléctrica.
#   2. Filtro Pasabanda Butterworth (20-150 Hz) — aísla la banda útil sEMG.
#
# Ambos filtros usan estado (`zi`) persistente entre llamadas, para evitar
# discontinuidades matemáticas al procesar muestra a muestra en tiempo
# real (en vez de procesar el audio/señal completa de una sola vez).
#
# INTERFAZ REQUERIDA por dsp.py (CapturadorVentanas):
#   - crear_filtros_por_canal() -> dict {nombre_canal: FiltroEMG}
#   - FiltroEMG.procesar(valor_float) -> float   (UNA muestra, no un chunk)
#   - FiltroEMG.reset() -> None
#
# NOTA: internamente, `lfilter` puede procesar arrays de cualquier
# longitud (incluida longitud 1) sin perder el estado `zi`, así que
# procesar muestra por muestra es tan matemáticamente correcto como
# procesar en chunks — solo se envuelve el valor en un array de 1
# elemento y se extrae el resultado escalar de vuelta.
# =============================================================================

import numpy as np
from scipy.signal import butter, lfilter, lfilter_zi, iirnotch

from src.config import (
    NOMBRES_CANALES,
    FS_TOTAL,
    FILTRO_NOTCH_FREQ_HZ,
    FILTRO_NOTCH_Q,
    FILTRO_BANDPASS_LOW_HZ,
    FILTRO_BANDPASS_HIGH_HZ,
    FILTRO_BUTTERWORTH_ORDER,
)


class FiltroEMG:
    """
    Filtro digital en tiempo real para señales sEMG.
    Aplica secuencialmente:
    1. Filtro Notch (50 Hz) para eliminar la interferencia de la red eléctrica.
    2. Filtro Pasabanda Butterworth (20-150 Hz) para aislar la señal muscular útil.
    """
    def __init__(self, fs=1000.0, lowcut=FILTRO_BANDPASS_LOW_HZ, highcut=FILTRO_BANDPASS_HIGH_HZ,
                 notch_freq=FILTRO_NOTCH_FREQ_HZ, notch_q=FILTRO_NOTCH_Q, order=FILTRO_BUTTERWORTH_ORDER):
        self.fs = fs

        # --- 1. Configuración del Filtro Notch (50 Hz) ---
        # notch_q = 30 es un valor estándar clínico (hace que el corte sea muy estrecho)
        self.b_notch, self.a_notch = iirnotch(notch_freq, notch_q, fs)
        self.zi_notch = lfilter_zi(self.b_notch, self.a_notch)

        # --- 2. Configuración del Filtro Pasabanda (20 - 150 Hz) ---
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq
        self.b_band, self.a_band = butter(order, [low, high], btype='band')
        self.zi_band = lfilter_zi(self.b_band, self.a_band)

    def procesar_chunk(self, chunk):
        """
        Procesa un arreglo (chunk) de muestras entrantes.
        Mantiene el estado 'zi' para evitar discontinuidades matemáticas
        entre las ventanas de datos en tiempo real.
        """
        # 1. Aplicar Filtro Notch para matar el ruido de 50Hz
        notch_out, self.zi_notch = lfilter(
            self.b_notch, self.a_notch, chunk, zi=self.zi_notch
        )

        # 2. Aplicar Filtro Pasabanda para aislar la banda EMG útil
        band_out, self.zi_band = lfilter(
            self.b_band, self.a_band, notch_out, zi=self.zi_band
        )

        return band_out

    def procesar(self, valor: float) -> float:
        """Procesa UNA muestra individual (interfaz que espera dsp.py:
        CapturadorVentanas.procesar_trama() llama esto una vez por
        muestra cruda, no por chunk). Internamente reutiliza
        procesar_chunk() con un array de longitud 1."""
        salida = self.procesar_chunk(np.array([valor], dtype=np.float64))
        return float(salida[0])

    def reset_state(self):
        """Limpia la memoria del filtro (útil al iniciar una nueva calibración)"""
        self.zi_notch = lfilter_zi(self.b_notch, self.a_notch)
        self.zi_band = lfilter_zi(self.b_band, self.a_band)

    def reset(self):
        """Alias de reset_state() — nombre que espera dsp.py
        (CapturadorVentanas.reset() llama f.reset() por cada filtro)."""
        self.reset_state()


def crear_filtros_por_canal(fs: float = FS_TOTAL, lowcut: float = FILTRO_BANDPASS_LOW_HZ,
                             highcut: float = FILTRO_BANDPASS_HIGH_HZ,
                             notch_freq: float = FILTRO_NOTCH_FREQ_HZ,
                             notch_q: float = FILTRO_NOTCH_Q,
                             order: int = FILTRO_BUTTERWORTH_ORDER) -> dict:
    """Crea un FiltroEMG independiente por cada canal (bíceps, tríceps,
    antebrazo), con estado 'zi' propio — necesario porque cada canal
    tiene su propia señal continua y no deben mezclar su historial de
    filtrado entre sí.

    NOTA sobre `fs`: dsp.py y config.py usan FS_TOTAL (1000 Hz, la tasa
    total del ADC), no FS (≈333 Hz, la tasa efectiva por canal). Esto es
    correcto aquí: cada canal recibe una muestra nueva cada vez que el
    Timer1 del Arduino le toca su turno en el ciclo de 3 canales, pero
    la propia señal EMG y el ruido de red (50Hz) existen en tiempo real
    continuo, así que el filtro debe diseñarse con la frecuencia de
    muestreo real a la que EFECTIVAMENTE llegan las muestras de ESE
    canal. Si se observa que el notch o el pasabanda no filtran bien en
    la práctica, revisar si conviene usar FS (≈333 Hz) en su lugar —
    ver la nota de Nyquist efectivo en config.py (NYQUIST_EFECTIVO_HZ).
    """
    return {
        nombre: FiltroEMG(fs=fs, lowcut=lowcut, highcut=highcut,
                           notch_freq=notch_freq, notch_q=notch_q, order=order)
        for nombre in NOMBRES_CANALES
    }