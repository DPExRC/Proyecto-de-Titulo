# =============================================================================
# main.py — Orquestador de producción: PC offload, regresor multi-salida
# =============================================================================
# Arquitectura (v3.0, confirmada):
#   - 3 canales sEMG: bíceps braquial, tríceps braquial, pronator teres (antebrazo)
#   - 2 DOF: codo (bidireccional) + muñeca (unidireccional), reposo = 0°
#   - Sin clasificador: gating de reposo/ruido por UMBRAL_BAJO (%MVC),
#     filtro exponencial asimétrico y slew-rate — responsabilidad del
#     firmware Arduino, no de este módulo.
#
# Hilos:
#   t_serial  → lee muestras crudas → aplica DSP → deposita features en cola
#   t_control → extrae features → corre regresor → envía 2 ángulos al Arduino
#
# Log en consola (cada ciclo donde algún ángulo varía > LOG_UMBRAL_CAMBIO°):
#   bic=67.3  tri=8.1  delt=22.4  →  codo=142.3°  muneca=21.0°
#
# Nota de estado: las features son RMS crudo, aún sin normalizar por %MVC
# (módulo de calibración pendiente). Los valores en el log son unidades
# ADC filtradas, no porcentaje de contracción.
# =============================================================================

import threading
import queue
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from src.config import (PORT, BAUDRATE, INTERVALO_CONTROL, NOMBRES_FEATURES)
from src.core.serial_bridge import SerialBridge
# predictor.py está actualmente en models/predictor.py — debería moverse
# a src/inference/predictor.py para separar código fuente de artefactos
# de modelo (.pkl). Actualizar este import si se reubica el archivo.
from models.predictor import EMGPredictor

# Umbral de variación mínima para emitir una línea de log (°)
LOG_UMBRAL_CAMBIO = 2.0

# Índices de RMS por canal en el vector de 12 features — resueltos por
# nombre para no romperse si el orden cambia en config.py
_IDX_RMS_BICEPS    = NOMBRES_FEATURES.index("rms_biceps")
_IDX_RMS_TRICEPS   = NOMBRES_FEATURES.index("rms_triceps")
_IDX_RMS_ANTEBRAZO = NOMBRES_FEATURES.index("rms_antebrazo")

cola_features: queue.Queue = queue.Queue(maxsize=10)
flag_activo = threading.Event()


# ------------------------------------------------------------------------------
def hilo_control(bridge: SerialBridge, predictor: EMGPredictor):
    angulo_codo_ant   = -1.0
    angulo_muneca_ant = -1.0

    while flag_activo.is_set():
        try:
            features = cola_features.get(timeout=1.0)
        except queue.Empty:
            continue

        t0 = time.time()

        resultado = predictor.predecir_angulos(features)
        angulo_codo   = resultado["angulo_codo"]
        angulo_muneca = resultado["angulo_muneca"]

        # Enviar ambos ángulos al Arduino
        bridge.enviar_angulos(angulo_codo, angulo_muneca)

        # Log cuando algún ángulo varía significativamente
        cambio_codo   = abs(angulo_codo   - angulo_codo_ant)   > LOG_UMBRAL_CAMBIO
        cambio_muneca = abs(angulo_muneca - angulo_muneca_ant) > LOG_UMBRAL_CAMBIO

        if cambio_codo or cambio_muneca:
            bic  = features[_IDX_RMS_BICEPS]
            tri  = features[_IDX_RMS_TRICEPS]
            delt = features[_IDX_RMS_ANTEBRAZO]
            print(
                f"  bic={bic:6.1f}  tri={tri:6.1f}  delt={delt:6.1f}"
                f"  →  codo={angulo_codo:6.1f}°  muneca={angulo_muneca:6.1f}°"
            )
            angulo_codo_ant   = angulo_codo
            angulo_muneca_ant = angulo_muneca

        # Respetar cadencia de control: si el procesamiento fue más rápido
        # que INTERVALO_CONTROL, esperar el tiempo restante
        transcurrido = time.time() - t0
        espera = INTERVALO_CONTROL - transcurrido
        if espera > 0:
            time.sleep(espera)


# ------------------------------------------------------------------------------
def main():
    print("=" * 60)
    print("  Sistema EMG — PC Offload  |  Regresor multi-salida  |  v3.0")
    print("  Canales: bíceps, tríceps, antebrazo")
    print("  DOF: codo (bidireccional) + muñeca (unidireccional)")
    print("  Reposo = 0° en ambos DOF")
    print("=" * 60)

    predictor = EMGPredictor()
    bridge    = SerialBridge(PORT, BAUDRATE)

    if not bridge.conectar():
        print("[main] No se pudo conectar al Arduino. "
              "Verifica PORT en src/config.py.")
        sys.exit(1)

    flag_activo.set()

    t_serial = threading.Thread(
        target=bridge.leer_muestras,
        args=(cola_features, flag_activo),
        daemon=True,
        name="t_serial"
    )
    t_ctrl = threading.Thread(
        target=hilo_control,
        args=(bridge, predictor),
        daemon=True,
        name="t_control"
    )

    t_serial.start()
    t_ctrl.start()

    print("[main] Sistema activo. Ctrl+C para detener.\n")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n[main] Deteniendo...")

    flag_activo.clear()
    time.sleep(0.5)          # dar tiempo a los hilos de terminar su ciclo actual
    bridge.desconectar()
    print("[main] Detenido.")


if __name__ == "__main__":
    main()