import sys
from pathlib import Path


from emg_arm.communication.serial_bridge import parsear_trama_emg, procesar_trama_emg


class DummyCapturador:
    def __init__(self):
        self.ultimas_valores = None

    def procesar_trama(self, valores):
        self.ultimas_valores = valores
        return [1.0, 2.0, 3.0]


class DummyCalibrador:
    def normalizar(self, vector):
        return [v * 10 for v in vector]


def test_parsear_trama_emg_valida_y_extrae_numeros():
    assert parsear_trama_emg("S,1.5,2.5,3.5") == [1.5, 2.5, 3.5]


def test_parsear_trama_emg_rechaza_tramas_invalidas():
    assert parsear_trama_emg("X,1,2,3") is None
    assert parsear_trama_emg("S,1,2") is None


def test_procesar_trama_emg_aplica_dsp_y_calibracion():
    capturador = DummyCapturador()
    calibrador = DummyCalibrador()

    vector = procesar_trama_emg([1.0, 2.0, 3.0], capturador, calibrador)

    assert vector == [10.0, 20.0, 30.0]
    assert capturador.ultimas_valores == [1.0, 2.0, 3.0]
