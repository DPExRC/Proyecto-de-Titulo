# emg_arm/communication/protocol.py
from typing import Optional
from emg_arm.config import NOMBRES_CANALES

PROTOCOLO_PREFIJO_RX = "S,"   # Arduino -> PC (muestras crudas)
PROTOCOLO_PREFIJO_TX = "A,"   # PC -> Arduino (ángulos objetivo)
PROTOCOLO_ACK = "A"           # Arduino -> PC (echo de comando procesado)

_ADC_MIN = 0
_ADC_MAX = 1023


def validar_rangos_adc(valores: list[float]) -> Optional[list[float]]:
    """Verifica que todos los valores ADC estén dentro del rango [0, 1023]."""
    for v in valores:
        if not (_ADC_MIN <= v <= _ADC_MAX):
            return None
    return valores


def parsear_trama_emg(linea: str) -> Optional[list[float]]:
    """Parsea una trama EMG cruda 'S,<v1>,<v2>,<v3>' sin depender del puerto serie."""
    if not linea.startswith(PROTOCOLO_PREFIJO_RX):
        return None
    partes = linea[len(PROTOCOLO_PREFIJO_RX):].split(",")
    if len(partes) != len(NOMBRES_CANALES):
        return None
    try:
        valores = [float(p) for p in partes]
    except ValueError:
        return None
    return validar_rangos_adc(valores)