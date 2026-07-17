import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from verification import (
    verificar_parseo_serial,
    verificar_filtro_discrimina_bandas,
    verificar_features_validas,
    verificar_predictor_responde_a_activacion,
    medir_latencia_pipeline,
)


# =============================================================================
# 1. Parseo serial — camino feliz + 3 casos que deben rechazarse
# =============================================================================
def test_parseo_serial_acepta_trama_valida():
    resultado = verificar_parseo_serial()
    assert resultado["trama_valida"] == [512.0, 480.0, 300.0]


def test_parseo_serial_rechaza_prefijo_invalido():
    resultado = verificar_parseo_serial()
    assert resultado["rechaza_prefijo_invalido"] is True


def test_parseo_serial_rechaza_canales_faltantes():
    resultado = verificar_parseo_serial()
    assert resultado["rechaza_canales_faltantes"] is True


def test_parseo_serial_rechaza_valor_no_numerico():
    # Cubre el branch try/except ValueError que no tenía test propio
    resultado = verificar_parseo_serial()
    assert resultado["rechaza_valor_no_numerico"] is True


# =============================================================================
# 2. Filtro — debe discriminar banda pasante de fuera de banda
# =============================================================================
def test_filtro_deja_pasar_señal_dentro_de_banda_con_poca_atenuacion():
    resultado = verificar_filtro_discrimina_bandas()
    # -3dB ~ 70% de la amplitud original; margen razonable para un
    # Butterworth orden 4 con transitorio ya descartado
    assert resultado["atenuacion_dentro_db"] > -3.0


def test_filtro_atenua_fuertemente_señal_fuera_de_banda():
    resultado = verificar_filtro_discrimina_bandas()
    assert resultado["atenuacion_fuera_db"] < -10.0


def test_filtro_discrimina_claramente_entre_bandas():
    # La diferencia entre ambas atenuaciones es lo que realmente prueba
    # que el filtro FILTRA algo, y no que ambas señales pasan igual
    # (que sería el caso de un filtro roto tipo passthrough)
    resultado = verificar_filtro_discrimina_bandas()
    diferencia_db = resultado["atenuacion_dentro_db"] - resultado["atenuacion_fuera_db"]
    assert diferencia_db > 10.0


# =============================================================================
# 3. Features — límites físicos reales, no solo "es finito"
# =============================================================================
def test_features_tiene_longitud_esperada():
    resultado = verificar_features_validas()
    assert resultado["longitud"] == 12


def test_features_son_finitas():
    resultado = verificar_features_validas()
    assert resultado["todos_finitos"] is True


def test_features_rms_mav_wl_no_negativos():
    resultado = verificar_features_validas()
    assert resultado["rms_no_negativo"] is True
    assert resultado["mav_no_negativo"] is True
    assert resultado["wl_no_negativo"] is True


def test_features_zc_en_rango_fisico():
    resultado = verificar_features_validas()
    assert resultado["zc_en_rango_fisico"] is True


# =============================================================================
# 4. Predictor — debe responder a la activación, no solo "estar en rango"
# =============================================================================
def test_predictor_flexiona_codo_ante_activacion_de_biceps():
    resultado = verificar_predictor_responde_a_activacion()
    # Umbral de 10° para tolerar diferencias entre modelo entrenado real
    # y el fallback proporcional — ambos deben mostrar una respuesta
    # clara en la misma dirección (bíceps activo -> más flexión)
    assert resultado["diferencia"] > 10.0


# =============================================================================
# 5. Latencia — medida real con time.perf_counter(), no un valor fijo
# =============================================================================
def test_latencia_pipeline_dentro_de_presupuesto():
    resultado = medir_latencia_pipeline(n_repeticiones=50)
    # Espiral 2 del plan de pruebas: procesamiento en host < 50ms
    assert resultado["promedio_ms"] < 50.0


def test_latencia_maxima_no_tiene_picos_excesivos():
    resultado = medir_latencia_pipeline(n_repeticiones=50)
    # El máximo puede ser más laxo que el promedio (permite algún pico
    # ocasional por garbage collection, etc.) pero no debe dispararse
    assert resultado["maximo_ms"] < 100.0
