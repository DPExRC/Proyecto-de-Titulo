# =============================================================================
# captura.py — Captura de dataset EMG, DSP completo en Python
# =============================================================================
# Uso:
#   python data/captura.py --port COM3
#   python data/captura.py --port /dev/ttyUSB0 --duracion 5
#
# Arquitectura (v3.0, confirmada): el Arduino transmite únicamente
# muestras crudas de los 3 canales sEMG, sincronizadas por trama. Este
# script aplica en Python el filtro IIR (filtro.py), el ventaneo
# deslizante y la extracción de características (features.py) — el
# Arduino no calcula RMS, MAV, WL ni ZC.
#
# Protocolo serial esperado (REQUIERE actualizar emg_v3.ino para que lo
# implemente — no está hecho todavía):
#   Cada trama es una línea de texto:
#     "S,<adc_biceps>,<adc_triceps>,<adc_deltoides>\n"
#   con valores enteros 0–1023 (lectura cruda del ADC de 10 bits), una
#   trama por ciclo de muestreo sincronizado (~333 Hz, los 3 canales
#   leídos en sucesión dentro del mismo ciclo de Timer1). Si el firmware
#   real usa otro formato u otra cadencia, ajustar PROTOCOLO_PREFIJO y
#   _parsear_trama() más abajo antes de capturar datos reales.
#
# Estado de la normalización %MVC: NO IMPLEMENTADA todavía (decisión
# explícita: se difiere). Las features que se guardan en el CSV son
# RMS/MAV/WL/ZC crudos sobre la señal filtrada, sin normalizar por
# baseline/MVC de sesión. Esto es aceptable para entrenar un primer
# modelo, pero limita la generalización entre sesiones y usuarios
# distintos (ver Capítulo 1, sección sobre normalización %MVC) hasta
# que se implemente el módulo de calibración.
#
# Flujo por captura:
#   1. Usuario posiciona el brazo en un ángulo de codo y de hombro
#      conocidos (puede dejar uno fijo en 0° y mover solo el otro).
#   2. Escribe ambos ángulos por teclado.
#   3. El script lee tramas crudas durante --duracion segundos, filtra
#      cada canal, ventanea y extrae el vector de 12 features cada
#      N_PASO muestras.
#   4. Guarda en CSV: NOMBRES_FEATURES (12 columnas) + angulo_codo +
#      angulo_hombro.
#
# El CSV acumula capturas entre ejecuciones (append).
# Convención de ángulos (fijada para todo el proyecto): reposo = 0° en
# ambos DOF. No usar 90° ni 180° como reposo.
# =============================================================================

import serial
import time
import csv
import os
import argparse
import sys
from collections import deque

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (BAUDRATE, DATA_PATH, DURACION_CAPTURA_S,
                         NOMBRES_FEATURES, NOMBRES_CANALES,
                         COLS_TARGET, COL_ANGULO_CODO, COL_ANGULO_HOMBRO,
                         N_VENTANA, N_PASO, ANGULO_MIN, ANGULO_MAX)
from processing.filtro import crear_filtros_por_canal
from processing.features import extraer_vector_features

PROTOCOLO_PREFIJO = "S,"   # trama cruda: "S,<v_biceps>,<v_triceps>,<v_deltoides>"
COLUMNAS_CSV = NOMBRES_FEATURES + COLS_TARGET


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
    """Consume líneas hasta detectar fin de calibración o agotar el timeout.
    Nota: si la calibración baseline/MVC se termina implementando en
    Python (módulo aún no construido, ver cabecera), esta función deja
    de tener sentido tal como está y debe revisarse."""
    print("[captura] Esperando fin de inicialización del firmware...")
    t0 = time.time()
    while time.time() - t0 < timeout:
        linea = ser.readline().decode("utf-8", errors="ignore").strip()
        if linea.startswith("#"):
            print(f"[firmware] {linea}")
        if "LISTO" in linea or "READY" in linea.upper():
            time.sleep(0.5)
            return
    print("[captura] Timeout esperando inicialización — continuando de todas formas.")


def _parsear_trama(linea: str):
    """Parsea una trama 'S,<v0>,<v1>,<v2>'. Retorna lista de 3 floats
    (orden: biceps, triceps, deltoides) o None si la línea es inválida."""
    if not linea.startswith(PROTOCOLO_PREFIJO):
        return None
    partes = linea[len(PROTOCOLO_PREFIJO):].split(",")
    if len(partes) != len(NOMBRES_CANALES):
        return None
    try:
        return [float(p) for p in partes]
    except ValueError:
        return None


class CapturadorVentanas:
    """Mantiene el estado de filtrado y ventaneo por canal, y emite un
    vector de 12 features cada vez que se acumulan N_PASO muestras
    nuevas sobre una ventana llena de N_VENTANA muestras filtradas."""

    def __init__(self):
        self.filtros = crear_filtros_por_canal()  # {nombre: FiltroCanal}
        self.buffers = {n: deque(maxlen=N_VENTANA) for n in NOMBRES_CANALES}
        self.contador_pasos = 0

    def reset(self):
        for f in self.filtros.values():
            f.reset()
        for b in self.buffers.values():
            b.clear()
        self.contador_pasos = 0

    def procesar_trama(self, valores_crudos: list):
        """Procesa una trama de 3 muestras crudas (una por canal).
        Retorna el vector de 12 features si se completó un paso de
        ventaneo, o None si aún no hay suficientes muestras."""
        for nombre, crudo in zip(NOMBRES_CANALES, valores_crudos):
            filtrada = self.filtros[nombre].procesar(crudo)
            self.buffers[nombre].append(filtrada)

        self.contador_pasos += 1
        ventana_llena = all(len(self.buffers[n]) == N_VENTANA for n in NOMBRES_CANALES)

        if ventana_llena and self.contador_pasos >= N_PASO:
            self.contador_pasos = 0
            ventanas = [list(self.buffers[n]) for n in NOMBRES_CANALES]
            return extraer_vector_features(ventanas)

        return None


def capturar_angulos(ser: serial.Serial, angulo_codo: float, angulo_hombro: float,
                      duracion_s: float, capturador: CapturadorVentanas) -> list:
    """Lee tramas crudas durante duracion_s segundos, filtra y ventanea
    en tiempo real. Devuelve lista de filas [12 features..., angulo_codo,
    angulo_hombro]."""
    registros = []
    t0 = time.time()
    t_ultimo_aviso = t0

    while time.time() - t0 < duracion_s:
        raw = ser.readline()
        if not raw:
            continue
        linea = raw.decode("utf-8", errors="ignore").strip()

        valores = _parsear_trama(linea)
        if valores is not None:
            vector_features = capturador.procesar_trama(valores)
            if vector_features is not None:
                registros.append(vector_features + [angulo_codo, angulo_hombro])

        ahora = time.time()
        if ahora - t_ultimo_aviso >= 1.0:
            restante = duracion_s - (ahora - t0)
            print(f"  [{restante:.0f}s restantes] {len(registros)} vectores", end="\r")
            t_ultimo_aviso = ahora

    print()
    return registros


def leer_angulos() -> tuple:
    """Solicita ambos ángulos objetivo por teclado. Retorna
    (angulo_codo, angulo_hombro) o (None, None) si el usuario escribe 'q'."""
    while True:
        entrada = input(
            f"\n  Ángulos a capturar — codo,hombro (ej: '90,0'), "
            f"reposo='0,0', o 'q' para terminar: "
        ).strip()
        if entrada.lower() == "q":
            return None, None
        try:
            partes = [p.strip() for p in entrada.split(",")]
            if len(partes) != 2:
                print("  Formato esperado: <angulo_codo>,<angulo_hombro> (ej: '90,0')")
                continue
            codo, hombro = float(partes[0]), float(partes[1])
            if not (ANGULO_MIN <= codo <= ANGULO_MAX):
                print(f"  angulo_codo fuera de [{ANGULO_MIN}, {ANGULO_MAX}].")
                continue
            if not (ANGULO_MIN <= hombro <= ANGULO_MAX):
                print(f"  angulo_hombro fuera de [{ANGULO_MIN}, {ANGULO_MAX}].")
                continue
            return codo, hombro
        except ValueError:
            print("  Entrada no válida. Formato: <angulo_codo>,<angulo_hombro>")


def resumen_csv(path: str):
    """Muestra distribución de combinaciones de ángulos registradas."""
    if not os.path.exists(path):
        return
    combinaciones = {}
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for fila in reader:
            try:
                key = (float(fila[COL_ANGULO_CODO]), float(fila[COL_ANGULO_HOMBRO]))
                combinaciones[key] = combinaciones.get(key, 0) + 1
            except (ValueError, KeyError):
                pass
    if combinaciones:
        print("\n[captura] Distribución actual del dataset (codo°, hombro°):")
        for (codo, hombro) in sorted(combinaciones):
            print(f"  ({codo:6.1f}°, {hombro:6.1f}°)  →  {combinaciones[(codo, hombro)]:4d} vectores")


# ------------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Captura dataset EMG — DSP en Python")
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--duracion", type=int, default=DURACION_CAPTURA_S,
                        help="Segundos de captura por combinación de ángulos")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(DATA_PATH)), exist_ok=True)
    archivo_nuevo = not os.path.exists(DATA_PATH)

    print("=" * 60)
    print("  Captura de Dataset EMG — DSP completo en Python")
    print(f"  Puerto: {args.port}  |  {args.duracion}s por captura")
    print(f"  Canales: {NOMBRES_CANALES}")
    print(f"  Convención: reposo = 0° en ambos DOF (codo y hombro)")
    print("  NOTA: features sin normalizar %MVC (módulo de calibración")
    print("        pendiente — ver cabecera de este archivo)")
    print("=" * 60)

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

    capturador = CapturadorVentanas()
    total = 0

    with open(DATA_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if archivo_nuevo:
            writer.writerow(COLUMNAS_CSV)

        print("\n[captura] Sistema listo. Posiciona el brazo en los ángulos")
        print("          deseados y mantén la posición durante la captura.")

        while True:
            angulo_codo, angulo_hombro = leer_angulos()
            if angulo_codo is None:
                break

            print(f"\n  Prepárate para mantener codo={angulo_codo:.0f}°, "
                  f"hombro={angulo_hombro:.0f}° durante {args.duracion}s...")
            for s in range(3, 0, -1):
                print(f"  {s}...", end="\r")
                time.sleep(1.0)
            print(f"  ¡CAPTURANDO codo={angulo_codo:.0f}°, hombro={angulo_hombro:.0f}°!   ")

            capturador.reset()  # evita arrastrar transitorios del filtro
            registros = capturar_angulos(ser, angulo_codo, angulo_hombro,
                                          args.duracion, capturador)

            for reg in registros:
                writer.writerow([f"{v:.6f}" for v in reg])
            f.flush()

            total += len(registros)
            print(f"  ✓ {len(registros)} vectores guardados "
                  f"(codo={angulo_codo:.0f}°, hombro={angulo_hombro:.0f}°)  "
                  f"(total acumulado: {total})")

            continuar = input("  ¿Capturar otra combinación? [Enter=sí / q=salir]: ").strip().lower()
            if continuar == "q":
                break

    ser.close()
    print(f"\n{'='*60}")
    print(f"  Dataset: {os.path.abspath(DATA_PATH)}")
    print(f"  Vectores añadidos en esta sesión: {total}")
    resumen_csv(DATA_PATH)
    print(f"{'='*60}")


if __name__ == "__main__":
    main()