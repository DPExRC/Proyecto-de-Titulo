# =============================================================================
# main.py — Punto de entrada único del proyecto EMG (menú + entrenamiento)
# =============================================================================
# Responsabilidades (SRP):
#   1. Mostrar el menú principal.
#   2. Orquestar el flujo ENTRENAR (calibración → captura → estandarización
#      → entrenamiento del modelo).
#   3. Delegar el flujo USAR (control en tiempo real) a control_loop.py.
#
# NO contiene: lógica de control en tiempo real, threading, paneles en vivo,
# evaluación de trayectoria. Eso vive en control_loop.py.
# =============================================================================

import os
import sys
import time
import serial

sys.path.insert(0, os.path.dirname(__file__))

from emg_arm.config import PORT, BAUDRATE, DATA_PATH, RUTA_CALIBRACION_DEFAULT
from emg_arm.processing.calibration import CalibradorEMG
from emg_arm.processing.dsp import CapturadorVentanas
from emg_arm.processing.standardization import (
    cargar_calibracion, normalizar_dataframe, reportar_saturaciones,
)
from training.train_model import entrenar_pipeline, DATA_PATH_NORMALIZADO
from data.capture import (
    esperar_ready, ejecutar_captura_interactiva,
)
from emg_arm.core.control_loop import flujo_usar

import pandas as pd

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich import box

console = Console()

RUTA_DATOS_NORMALIZADOS = DATA_PATH_NORMALIZADO


# =============================================================================
# Helpers de conexión serial
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
        time.sleep(2.0)
        ok = esperar_ready(ser)

    if not ok:
        console.print("[bold red]✗ READY no recibido.[/] Verifica firmware y puerto.")
        ser.close()
        return None

    ser.reset_input_buffer()
    console.print(f"[bold green]✓[/] Firmware conectado en [cyan]{puerto}[/].")
    return ser


# =============================================================================
# MODO 1 — ENTRENAR
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

    # --- 2. Captura de dataset ------------------------------------------------
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

    # --- 3. Estandarización (offline) ------------------------------------------
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

    # --- 4. Entrenamiento (offline) --------------------------------------------
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
# MENÚ PRINCIPAL
# =============================================================================
def run():
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
    run()