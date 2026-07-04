# =============================================================================
# main.py — Punto de entrada único del proyecto EMG (con salida via rich)
# =============================================================================
# Unifica en un menú los dos modos de uso del sistema:
#
#   1) ENTRENAR  — calibración (baseline/MVC) + captura de dataset +
#                  estandarización (%MVC) + entrenamiento del regresor.
#   2) USAR      — carga calibración y modelo ya entrenados, conecta el
#                  Arduino, y corre el loop de inferencia en tiempo real.
#
# Requiere: pip install rich
#
# Cada etapa sigue viviendo en su propio módulo (calibracion.py,
# captura.py, estandarizacion.py, train.py, serial_bridge.py,
# predictor.py) — este archivo solo orquesta y presenta.
# =============================================================================

import os
import sys
import time
import threading
import queue
import serial

sys.path.insert(0, os.path.dirname(__file__))

from src.config import PORT, BAUDRATE, INTERVALO_CONTROL, NOMBRES_FEATURES, DATA_PATH
from src.processing.calibration import CalibradorEMG, RUTA_CALIBRACION_DEFAULT
from src.processing.dsp import CapturadorVentanas
from src.processing.standardization import (
    cargar_calibracion, normalizar_dataframe, reportar_saturaciones,
)
from training.train_model import entrenar_pipeline, DATA_PATH_NORMALIZADO
from data.capture import esperar_ready, ejecutar_captura_interactiva
from src.core.serial_bridge import SerialBridge
from models.predictor import EMGPredictor

import pandas as pd

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.live import Live
from rich import box

console = Console()

RUTA_DATOS_NORMALIZADOS = DATA_PATH_NORMALIZADO  # misma ruta que usa train.py

LOG_UMBRAL_CAMBIO = 2.0

_IDX_RMS_BICEPS    = NOMBRES_FEATURES.index("rms_biceps")
_IDX_RMS_TRICEPS   = NOMBRES_FEATURES.index("rms_triceps")
_IDX_RMS_ANTEBRAZO = NOMBRES_FEATURES.index("rms_antebrazo")

cola_features: queue.Queue = queue.Queue(maxsize=10)
flag_activo = threading.Event()


# =============================================================================
# Helpers de conexión serial (comunes a ambos flujos)
# =============================================================================
def _conectar_y_esperar_ready(puerto: str):
    """Abre el puerto, espera el READY del firmware con un spinner, y
    limpia el buffer. Retorna el objeto Serial o None si falló."""
    try:
        ser = serial.Serial(puerto, BAUDRATE, timeout=2.0)
    except serial.SerialException as e:
        console.print(f"[bold red]✗ Error al abrir puerto {puerto}:[/] {e}")
        return None

    with console.status("[cyan]Esperando READY del firmware...", spinner="dots"):
        time.sleep(2.0)  # deja llegar el READY antes de limpiar el buffer
        ok = esperar_ready(ser)

    if not ok:
        console.print("[bold red]✗ READY no recibido.[/] Verifica firmware y puerto.")
        ser.close()
        return None

    ser.reset_input_buffer()
    console.print(f"[bold green]✓[/] Firmware conectado en [cyan]{puerto}[/].")
    return ser


# =============================================================================
# MODO 1 — ENTRENAR: calibrar + capturar + estandarizar + entrenar
# =============================================================================
def flujo_entrenar():
    console.print(Panel.fit(
        "[bold]Calibración[/] → [bold]Captura[/] → [bold]Estandarización[/] → [bold]Entrenamiento[/]",
        title="[bold cyan]MODO ENTRENAR[/]", border_style="cyan"
    ))

    puerto = Prompt.ask("  Puerto serial", default=PORT)
    ser = _conectar_y_esperar_ready(puerto)
    if ser is None:
        return

    # --- 1. Calibración ------------------------------------------------------
    console.rule("[bold yellow]Paso 1/4 — Calibración baseline/MVC")
    calibrador = CalibradorEMG()
    capturador_calib = CapturadorVentanas()
    calibrador.ejecutar(ser, capturador_calib)
    calibrador.guardar(RUTA_CALIBRACION_DEFAULT)
    console.print(f"[bold green]✓[/] Calibración guardada en "
                  f"[cyan]{RUTA_CALIBRACION_DEFAULT}[/]\n")

    # --- 2. Captura de dataset -------------------------------------------------
    console.rule("[bold yellow]Paso 2/4 — Captura de dataset")
    if not Confirm.ask("  ¿Listo para comenzar la captura?", default=True):
        ser.close()
        console.print("[yellow]Captura cancelada por el usuario.[/]")
        return

    total_capturado = ejecutar_captura_interactiva(ser, ruta_salida=DATA_PATH)
    ser.close()

    if total_capturado == 0:
        console.print("[bold red]✗ No se capturó ningún vector.[/] "
                       "Abortando estandarización y entrenamiento.")
        return
    console.print(f"[bold green]✓[/] {total_capturado} vectores capturados.\n")

    # --- 3. Estandarización (offline, sin serial) -------------------------------
    console.rule("[bold yellow]Paso 3/4 — Estandarización %MVC")
    with console.status("[cyan]Normalizando dataset...", spinner="dots"):
        df = pd.read_csv(DATA_PATH)
        calibracion = cargar_calibracion(RUTA_CALIBRACION_DEFAULT)
        df_normalizado = normalizar_dataframe(df, calibracion)

    reportar_saturaciones(df_normalizado)

    os.makedirs(os.path.dirname(os.path.abspath(RUTA_DATOS_NORMALIZADOS)), exist_ok=True)
    df_normalizado.to_csv(RUTA_DATOS_NORMALIZADOS, index=False)
    console.print(f"[bold green]✓[/] Dataset normalizado guardado en "
                  f"[cyan]{RUTA_DATOS_NORMALIZADOS}[/]\n")

    # --- 4. Entrenamiento (offline, sin serial) ----------------------------------
    console.rule("[bold yellow]Paso 4/4 — Entrenamiento del modelo")
    try:
        with console.status("[cyan]Entrenando RandomForestRegressor "
                             "(esto puede tardar unos segundos)...", spinner="dots"):
            _, meta = entrenar_pipeline(csv_path=RUTA_DATOS_NORMALIZADOS)
    except Exception as e:
        console.print(f"[bold red]✗ Error durante el entrenamiento:[/] {e}")
        return

    _mostrar_tabla_metricas(meta["regresor"]["test"])

    console.print(Panel.fit(
        "[bold green]Pipeline completo.[/] Selecciona [bold]'Usar'[/] en el "
        "menú para mover el brazo con este modelo.",
        border_style="green"
    ))


def _mostrar_tabla_metricas(test_metricas: dict):
    tabla = Table(title="Métricas de validación (hold-out)", box=box.ROUNDED)
    tabla.add_column("DOF", style="bold cyan")
    tabla.add_column("MAE (°)", justify="right")
    tabla.add_column("R²", justify="right")
    tabla.add_column("Estado", justify="center")

    for nombre, m in test_metricas.items():
        mae = m["test_mae"]
        r2 = m["test_r2"]
        if mae <= 10:
            estado = "[bold green]✓ Bueno[/]"
        elif mae <= 15:
            estado = "[bold yellow]~ Aceptable[/]"
        else:
            estado = "[bold red]✗ Revisar[/]"
        tabla.add_row(nombre, f"{mae:.2f}", f"{r2:.3f}", estado)

    console.print(tabla)


# =============================================================================
# MODO 2 — USAR: inferencia en tiempo real, mover el brazo
# =============================================================================
def hilo_control(bridge: SerialBridge, predictor: EMGPredictor, estado: dict):
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

        bridge.enviar_angulos(angulo_codo, angulo_muneca)

        # Estado compartido con el hilo principal para el panel en vivo
        estado["bic"]    = features[_IDX_RMS_BICEPS]
        estado["tri"]    = features[_IDX_RMS_TRICEPS]
        estado["delt"]   = features[_IDX_RMS_ANTEBRAZO]
        estado["codo"]   = angulo_codo
        estado["muneca"] = angulo_muneca
        estado["actualizaciones"] = estado.get("actualizaciones", 0) + 1

        cambio_codo   = abs(angulo_codo   - angulo_codo_ant)   > LOG_UMBRAL_CAMBIO
        cambio_muneca = abs(angulo_muneca - angulo_muneca_ant) > LOG_UMBRAL_CAMBIO
        if cambio_codo or cambio_muneca:
            angulo_codo_ant   = angulo_codo
            angulo_muneca_ant = angulo_muneca

        transcurrido = time.time() - t0
        espera = INTERVALO_CONTROL - transcurrido
        if espera > 0:
            time.sleep(espera)


def _panel_estado_en_vivo(estado: dict) -> Panel:
    tabla = Table.grid(padding=(0, 2))
    tabla.add_column(justify="right", style="bold")
    tabla.add_column()

    tabla.add_row("Bíceps (RMS):",    f"{estado.get('bic', 0):6.1f}")
    tabla.add_row("Tríceps (RMS):",   f"{estado.get('tri', 0):6.1f}")
    tabla.add_row("Antebrazo (RMS):", f"{estado.get('delt', 0):6.1f}")
    tabla.add_row("", "")
    tabla.add_row("Codo →",   f"[bold cyan]{estado.get('codo', 0):6.1f}°[/]")
    tabla.add_row("Muñeca →", f"[bold cyan]{estado.get('muneca', 0):6.1f}°[/]")
    tabla.add_row("", "")
    tabla.add_row("Actualizaciones:", str(estado.get("actualizaciones", 0)))

    return Panel(tabla, title="[bold green]Sistema activo — Ctrl+C para detener[/]",
                 border_style="green")


def flujo_usar():
    console.print(Panel.fit(
        "Carga el modelo y la calibración ya entrenados, y mueve el brazo "
        "en tiempo real.",
        title="[bold cyan]MODO USAR[/]", border_style="cyan"
    ))

    if not os.path.exists(RUTA_CALIBRACION_DEFAULT):
        console.print("[bold red]✗ No hay calibración guardada.[/] "
                       "Corre 'Entrenar' primero, o recalibra manualmente.")
        return

    try:
        with console.status("[cyan]Cargando modelo entrenado...", spinner="dots"):
            predictor = EMGPredictor()
    except Exception as e:
        console.print(f"[bold red]✗ No se pudo cargar el modelo entrenado:[/] {e}")
        console.print("  Corre [bold]'Entrenar'[/] primero.")
        return
    console.print("[bold green]✓[/] Modelo cargado.")

    puerto = Prompt.ask("  Puerto serial", default=PORT)
    bridge = SerialBridge(puerto, BAUDRATE)

    with console.status("[cyan]Conectando al Arduino...", spinner="dots"):
        conectado = bridge.conectar()

    if not conectado:
        console.print("[bold red]✗ No se pudo conectar al Arduino.[/]")
        return
    console.print(f"[bold green]✓[/] Arduino conectado en [cyan]{puerto}[/].")

    recalibrar = Confirm.ask(
        "  ¿Recalibrar baseline/MVC ahora? "
        "(recomendado si cambiaste los electrodos)", default=False
    )
    if recalibrar:
        calibrado = bridge.ejecutar_calibracion()
        if not calibrado:
            console.print("[yellow]⚠ Calibración no completada.[/] "
                           "Se usará la última calibración guardada, si existe.")
    else:
        console.print(f"[cyan]ℹ[/] Usando la última calibración guardada en "
                       f"{RUTA_CALIBRACION_DEFAULT}.")
        bridge.cargar_calibracion(RUTA_CALIBRACION_DEFAULT)

    estado = {}
    flag_activo.set()

    t_serial = threading.Thread(
        target=bridge.leer_muestras,
        args=(cola_features, flag_activo),
        daemon=True,
        name="t_serial"
    )
    t_ctrl = threading.Thread(
        target=hilo_control,
        args=(bridge, predictor, estado),
        daemon=True,
        name="t_control"
    )

    t_serial.start()
    t_ctrl.start()

    console.print()
    try:
        with Live(_panel_estado_en_vivo(estado), console=console,
                  refresh_per_second=8) as live:
            while True:
                live.update(_panel_estado_en_vivo(estado))
                time.sleep(0.1)
    except KeyboardInterrupt:
        console.print("\n[yellow]Deteniendo...[/]")

    flag_activo.clear()
    time.sleep(0.5)
    bridge.desconectar()
    console.print("[bold green]✓ Detenido.[/]")


# =============================================================================
# MENÚ PRINCIPAL
# =============================================================================
def main():
    console.print(Panel.fit(
        "[bold]Prótesis de brazo — 3 canales sEMG, 2 servos[/]\n"
        "bíceps · tríceps · antebrazo  →  codo · muñeca",
        title="[bold magenta]Sistema EMG[/]", border_style="magenta"
    ))

    while True:
        console.print()
        tabla = Table(show_header=False, box=box.SIMPLE)
        tabla.add_column(style="bold cyan", width=4)
        tabla.add_column()
        tabla.add_row("1)", "Entrenar  (calibrar + capturar + estandarizar + entrenar)")
        tabla.add_row("2)", "Usar      (mover el brazo con el modelo ya entrenado)")
        tabla.add_row("3)", "Salir")
        console.print(tabla)

        opcion = Prompt.ask("  Selecciona una opción", choices=["1", "2", "3"], default="3")

        if opcion == "1":
            flujo_entrenar()
        elif opcion == "2":
            flujo_usar()
        elif opcion == "3":
            console.print("[bold]Hasta luego.[/]")
            break


if __name__ == "__main__":
    main()