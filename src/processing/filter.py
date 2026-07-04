# =============================================================================
# filter.py — Filtro IIR Butterworth pasabanda, con estado, por canal
# =============================================================================
# Reemplaza el filtrado que antes hacía el Arduino (versión de 1 canal).
# En la arquitectura confirmada, el Arduino solo transmite muestras
# crudas; este módulo aplica el filtro digital en Python, sobre cada
# canal de forma independiente.
#
# Los coeficientes se calculan en tiempo de ejecución a partir de los
# parámetros de config.py (FS, FILTRO_CORTE_HZ), en vez de hardcodearse
# como constantes sueltas — evita repetir el problema ya detectado en
# el informe de tener 2-3 valores de corte distintos circulando entre
# capítulos y archivos.
#
# Diseño: Butterworth orden 4, pasabanda [20, FILTRO_CORTE_HZ] Hz,
# implementado en secciones de segundo orden (SOS) por estabilidad
# numérica en uso de streaming en tiempo real (preferible a la forma
# directa I con coeficientes b/a sueltos, que es más sensible a errores
# de redondeo en filtros de orden alto).
# =============================================================================

import numpy as np
from scipy.signal import butter, sosfilt

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import FS, FILTRO_CORTE_HZ, NOMBRES_CANALES

FRECUENCIA_BAJA_HZ = 20.0  # límite inferior de la banda pasante, fijo


def _validar_nyquist():
    nyquist = FS / 2.0
    if FILTRO_CORTE_HZ >= nyquist:
        raise ValueError(
            f"FILTRO_CORTE_HZ ({FILTRO_CORTE_HZ} Hz) debe ser menor que "
            f"el Nyquist efectivo ({nyquist:.1f} Hz) derivado de FS "
            f"({FS:.2f} Hz/canal en config.py). Revisar N_CANALES o "
            f"FILTRO_CORTE_HZ."
        )


def diseñar_sos():
    """Calcula los coeficientes SOS del filtro, una sola vez."""
    _validar_nyquist()
    sos = butter(
        N=4,
        Wn=[FRECUENCIA_BAJA_HZ, FILTRO_CORTE_HZ],
        btype="bandpass",
        fs=FS,
        output="sos",
    )
    return sos


# Coeficientes compartidos por todas las instancias de FiltroCanal —
# se calculan una sola vez al importar el módulo.
SOS_GLOBAL = diseñar_sos()


class FiltroCanal:
    """Filtro IIR con estado para un único canal. Procesa una muestra
    a la vez, manteniendo el estado interno entre llamadas (necesario
    para uso en streaming en tiempo real, a diferencia de filtrar un
    arreglo completo de una sola vez)."""

    def __init__(self, nombre_canal: str = ""):
        self.nombre_canal = nombre_canal
        self.sos = SOS_GLOBAL
        # Estado inicial en cero (n_secciones, 2)
        self.zi = np.zeros((self.sos.shape[0], 2))

    def procesar(self, muestra: float) -> float:
        """Filtra una única muestra cruda y retorna el valor filtrado."""
        salida, self.zi = sosfilt(self.sos, [muestra], zi=self.zi)
        return float(salida[0])

    def reset(self):
        """Reinicia el estado del filtro (usar al iniciar una nueva
        sesión de calibración o captura, para no arrastrar transitorios
        de la sesión anterior)."""
        self.zi = np.zeros((self.sos.shape[0], 2))


def crear_filtros_por_canal() -> dict:
    """Crea un FiltroCanal independiente por cada canal definido en
    config.NOMBRES_CANALES. Retorna un dict {nombre_canal: FiltroCanal}."""
    return {nombre: FiltroCanal(nombre) for nombre in NOMBRES_CANALES}


if __name__ == "__main__":
    # Prueba de cordura: una sinusoide dentro de la banda pasante debe
    # salir con amplitud similar (ganancia ≈ 1 en banda de paso); una
    # sinusoide fuera de banda debe atenuarse fuertemente.
    t = np.arange(0, 1.0, 1.0 / FS)

    f_dentro = 60.0   # Hz, dentro de 20-150
    f_fuera  = 5.0     # Hz, fuera de banda (por debajo de 20 Hz)

    señal_dentro = 100.0 * np.sin(2 * np.pi * f_dentro * t)
    señal_fuera  = 100.0 * np.sin(2 * np.pi * f_fuera * t)

    filtro_a = FiltroCanal("test_dentro")
    filtro_b = FiltroCanal("test_fuera")

    salida_dentro = [filtro_a.procesar(x) for x in señal_dentro]
    salida_fuera  = [filtro_b.procesar(x) for x in señal_fuera]

    # Ignorar el transitorio inicial al medir amplitud
    amp_dentro = np.std(salida_dentro[len(salida_dentro)//3:])
    amp_fuera  = np.std(salida_fuera[len(salida_fuera)//3:])
    amp_entrada = 100.0 / np.sqrt(2)  # RMS teórico de la entrada

    print(f"FS efectivo por canal: {FS:.2f} Hz")
    print(f"Nyquist efectivo: {FS/2:.2f} Hz, corte filtro: {FILTRO_CORTE_HZ} Hz")
    print(f"Señal a {f_dentro} Hz (dentro de banda): RMS salida ≈ {amp_dentro:.1f} "
          f"(entrada RMS ≈ {amp_entrada:.1f}) — debe pasar con poca atenuación")
    print(f"Señal a {f_fuera} Hz (fuera de banda): RMS salida ≈ {amp_fuera:.1f} "
          f"(entrada RMS ≈ {amp_entrada:.1f}) — debe atenuarse fuertemente")