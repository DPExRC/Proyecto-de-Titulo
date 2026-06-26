# =============================================================================
# main.py — Orquestador de producción: PC offload con pipeline dual
# =============================================================================
# Hilos:
#   t_serial  → lee vectores F,... del Arduino → deposita en cola
#   t_control → clasifica + regresa ángulo → envía A,<angulo> al Arduino
#
# Log en consola (cada ciclo donde cambia clase o ángulo varía > 2°):
#   [FLEXION ]  bic=67.3%  tri=8.1%  →  142.3°
# =============================================================================

import threading
import queue
import time
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from config import PORT, BAUDRATE, INTERVALO_CONTROL, CLASES
from core.serial_bridge import SerialBridge
from models.predictor import EMGPredictor

cola_features: queue.Queue = queue.Queue(maxsize=10)
flag_activo = threading.Event()


def hilo_control(bridge: SerialBridge, predictor: EMGPredictor):
    t_ultimo      = time.time()
    clase_ant     = -1
    angulo_ant    = -1.0

    while flag_activo.is_set():
        try:
            features = cola_features.get(timeout=1.0)
        except queue.Empty:
            continue

        # Cadencia de control
        ahora  = time.time()
        espera = INTERVALO_CONTROL - (ahora - t_ultimo)
        if espera > 0:
            time.sleep(espera)
        t_ultimo = time.time()

        clase, angulo = predictor.predecir_angulo(features)

        # Enviar ángulo al Arduino (el Arduino aplica el limitador de tasa)
        bridge.enviar_angulo(angulo)

        # Log cuando cambia clase o ángulo varía más de 2°
        cambio_clase  = clase != clase_ant
        cambio_angulo = abs(angulo - angulo_ant) > 2.0

        if cambio_clase or cambio_angulo:
            nombre = predictor.nombre_clase(clase)
            print(f"  [{nombre:<9}]  bic={features[0]:5.1f}%  tri={features[2]:5.1f}%"
                  f"  zcr0={features[1]:.4f}  zcr1={features[3]:.4f}"
                  f"  →  {angulo:6.1f}°")
            clase_ant  = clase
            angulo_ant = angulo


def main():
    print("=" * 60)
    print("  Sistema EMG — PC Offload  |  Pipeline dual  |  v3.0")
    print("  Clasificador: REPOSO / FLEXION / EXTENSION")
    print("  Regresor:     ángulo continuo 0°–180°")
    print("=" * 60)

    predictor = EMGPredictor()
    bridge    = SerialBridge(PORT, BAUDRATE)

    if not bridge.conectar():
        print("[main] No se pudo conectar al Arduino. Verifica PORT en config.py")
        sys.exit(1)

    flag_activo.set()

    t_serial = threading.Thread(
        target=bridge.leer_features,
        args=(cola_features, flag_activo),
        daemon=True, name="t_serial"
    )
    t_ctrl = threading.Thread(
        target=hilo_control,
        args=(bridge, predictor),
        daemon=True, name="t_control"
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
    bridge.desconectar()
    print("[main] Detenido.")


if __name__ == "__main__":
    main()