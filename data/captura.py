# =============================================================================
# captura.py — Captura de dataset EMG etiquetado por ángulo declarado
# =============================================================================
# Uso:
#   python data/captura.py --port COM3
#   python data/captura.py --port /dev/ttyUSB0 --duracion 5
#
# Flujo por captura:
#   1. Usuario posiciona el brazo en un ángulo conocido
#   2. Escribe el ángulo (0–180°) por teclado y presiona Enter
#   3. El script captura vectores F,... durante --duracion segundos
#   4. Guarda en CSV: rms_biceps, zcr_biceps, rms_triceps, zcr_triceps, angulo
#
# El CSV acumula capturas entre ejecuciones (append).
# Convención de ángulos:
#   180° = brazo extendido / reposo
#    90° = flexión media
#     0° = flexión completa (hombro con muñeca)
# =============================================================================

import serial
import time
import csv
import os
import argparse
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (BAUDRATE, DATA_PATH, DURACION_CAPTURA_S, NOMBRES_FEATURES)

# Columnas del CSV para modo regresor (ángulo continuo)
COLUMNAS_CSV = NOMBRES_FEATURES + ["angulo"]


# ------------------------------------------------------------------------------
def esperar_ready(ser: serial.Serial, timeout: float = 30.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        linea = ser.readline().decode("utf-8", errors="ignore").strip()
        if linea == "READY":
            return True
        if linea.startswith("#") and linea:
            print(f"[firmware] {linea}")
    return False


def esperar_calibracion(ser: serial.Serial, timeout: float = 40.0):
    """
    Consume líneas del serial hasta detectar el fin de calibración
    o hasta agotar el timeout.
    """
    print("[captura] Esperando fin de calibración del firmware...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        linea = ser.readline().decode("utf-8", errors="ignore").strip()
        if linea.startswith("#"):
            print(f"[firmware] {linea}")
        if "CALIBRACIÓN COMPLETADA" in linea or "CALIBRACION COMPLETADA" in linea:
            print("[captura] Calibración completada.")
            time.sleep(1.0)
            return
    print("[captura] Timeout esperando calibración — continuando de todas formas.")


def capturar_angulo(ser: serial.Serial, angulo: float, duracion_s: float) -> list:
    """
    Captura vectores F,... durante duracion_s segundos.
    Devuelve lista de filas [rms0, zcr0, rms1, zcr1, angulo].
    """
    registros = []
    t0 = time.time()
    t_ultimo_aviso = t0

    while time.time() - t0 < duracion_s:
        raw = ser.readline()
        if not raw:
            continue
        linea = raw.decode("utf-8", errors="ignore").strip()

        if linea.startswith("F,"):
            partes = linea[2:].split(",")
            if len(partes) == 4:
                try:
                    features = [float(p) for p in partes]
                    features.append(float(angulo))
                    registros.append(features)
                except ValueError:
                    pass

        ahora = time.time()
        if ahora - t_ultimo_aviso >= 1.0:
            restante = duracion_s - (ahora - t0)
            print(f"  [{restante:.0f}s restantes] {len(registros)} vectores", end="\r")
            t_ultimo_aviso = ahora

    print()
    return registros


def leer_angulo() -> float | None:
    """
    Solicita al usuario un ángulo por teclado.
    Devuelve el float validado, o None si el usuario escribe 'q' para salir.
    """
    while True:
        entrada = input("\n  Ángulo a capturar (0–180°), o 'q' para terminar: ").strip()
        if entrada.lower() == "q":
            return None
        try:
            valor = float(entrada)
            if 0.0 <= valor <= 180.0:
                return valor
            print("  Valor fuera de rango. Ingresa un número entre 0 y 180.")
        except ValueError:
            print("  Entrada no válida. Ingresa un número (ej: 90) o 'q' para salir.")


def resumen_csv(path: str):
    """Muestra distribución de ángulos registrados en el CSV."""
    if not os.path.exists(path):
        return
    angulos = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for fila in reader:
            try:
                ang = float(fila["angulo"])
                angulos[ang] = angulos.get(ang, 0) + 1
            except (ValueError, KeyError):
                pass
    if angulos:
        print("\n[captura] Distribución actual del dataset:")
        for ang in sorted(angulos):
            print(f"  {ang:6.1f}°  →  {angulos[ang]:4d} vectores")


# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Captura dataset EMG por ángulo declarado")
    parser.add_argument("--port",     default="COM3")
    parser.add_argument("--duracion", type=int, default=DURACION_CAPTURA_S,
                        help="Segundos de captura por ángulo")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(DATA_PATH)), exist_ok=True)
    archivo_nuevo = not os.path.exists(DATA_PATH)

    print("=" * 55)
    print("  Captura de Dataset EMG — Modo ángulo declarado")
    print(f"  Puerto: {args.port}  |  {args.duracion}s por captura")
    print("  Convención: 180°=reposo  90°=flexión media  0°=flexión completa")
    print("=" * 55)

    try:
        ser = serial.Serial(args.port, BAUDRATE, timeout=2.0)
    except serial.SerialException as e:
        print(f"[captura] Error al abrir puerto: {e}")
        sys.exit(1)

    time.sleep(2.0)
    ser.reset_input_buffer()

    print("[captura] Esperando READY del firmware...")
    if not esperar_ready(ser):
        print("[captura] READY no recibido. Verifica firmware y puerto.")
        ser.close()
        sys.exit(1)

    esperar_calibracion(ser)
    ser.reset_input_buffer()

    total = 0

    with open(DATA_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if archivo_nuevo:
            writer.writerow(COLUMNAS_CSV)

        print("\n[captura] Sistema listo. Posiciona el brazo en el ángulo deseado,")
        print("          escribe el valor y mantén la posición durante la captura.")

        while True:
            angulo = leer_angulo()
            if angulo is None:
                break

            print(f"\n  Prepárate para mantener {angulo:.0f}° durante {args.duracion}s...")
            for s in range(3, 0, -1):
                print(f"  {s}...", end="\r")
                time.sleep(1.0)
            print(f"  ¡CAPTURANDO {angulo:.0f}°!   ")

            registros = capturar_angulo(ser, angulo, args.duracion)

            for reg in registros:
                writer.writerow([f"{v:.6f}" for v in reg])
            f.flush()

            total += len(registros)
            print(f"  ✓ {len(registros)} vectores guardados para {angulo:.0f}°  (total acumulado: {total})")

            continuar = input("  ¿Capturar otro ángulo? [Enter=sí / q=salir]: ").strip().lower()
            if continuar == "q":
                break

    ser.close()
    print(f"\n{'='*55}")
    print(f"  Dataset: {os.path.abspath(DATA_PATH)}")
    print(f"  Vectores añadidos en esta sesión: {total}")
    resumen_csv(DATA_PATH)
    print(f"{'='*55}")


if __name__ == "__main__":
    main()