# =============================================================================
# captura.py — Captura de dataset EMG, DSP completo en Python
# =============================================================================
# Uso independiente:
#   python data/captura.py --port COM5
#   python data/captura.py --port /dev/ttyUSB0 --duracion 5
#
# Uso desde main.py (flujo "Entrenar"):
#   from data.captura import ejecutar_captura_interactiva
#   ejecutar_captura_interactiva(ser)   # ser ya abierto y con READY recibido
#
# Protocolo serial: "S,<adc_biceps>,<adc_triceps>,<adc_antebrazo>\n"
# Guarda features CRUDOS (sin normalizar) + angulo_codo + angulo_muneca.
# La normalización %MVC se hace después, offline, en estandarizacion.py —
# ver la discusión de diseño en ese archivo sobre por qué no se normaliza
# en el momento de la captura.
# =============================================================================

# =============================================================================
# captura.py — Captura de dataset EMG, DSP completo en Python (Visual Rich)
# =============================================================================

import serial
import time
import csv
import os
import re
import json
import argparse
import sys
import uuid
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (BAUDRATE, DATA_PATH, DURACION_CAPTURA_S,
                         NOMBRES_FEATURES, NOMBRES_CANALES,
                         COLS_TARGET, COL_ANGULO_CODO, COL_ANGULO_MUNECA,
                         N_VENTANA, N_PASO, ANGULO_MIN, ANGULO_MAX, PORT)
from src.processing.dsp import CapturadorVentanas

# Importaciones de la estética unificada Rich
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn
from rich import box

console = Console()

PROTOCOLO_PREFIJO = "S,"   
# NUEVO: cada fila queda etiquetada con la sesión de captura que la generó
# (sesion_id + timestamp), para poder reconstruir la Tabla 6.6 (número de
# sesiones, repeticiones, duración total) sin depender de la memoria de
# quién capturó qué. Antes el CSV solo tenía features + ángulos.
COLUMNAS_CSV = ["sesion_id", "timestamp"] + NOMBRES_FEATURES + COLS_TARGET
PATRON_NUMERO = re.compile(r"^-?\d+(\.\d+)?$")

RUTA_SESIONES = os.path.join(os.path.dirname(__file__), "sesiones.json")


def generar_sesion_id() -> str:
    """ID de sesión legible y único: fecha + sufijo corto aleatorio.
    Ej: 20260710_143210_a1b2"""
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:4]}"


def _cargar_sesiones() -> list:
    if not os.path.exists(RUTA_SESIONES):
        return []
    try:
        with open(RUTA_SESIONES, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def registrar_sesion(registro: dict, ruta: str = RUTA_SESIONES) -> None:
    """Persiste UNA sesión más en data/sesiones.json (lista acumulativa,
    nunca se sobreescribe lo anterior). Esta es la fuente de verdad para
    llenar la Tabla 6.6 (composición del dataset)."""
    sesiones = _cargar_sesiones()
    sesiones.append(registro)
    os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
    with open(ruta, "w") as f:
        json.dump(sesiones, f, indent=2, ensure_ascii=False)


# ------------------------------------------------------------------------------
def esperar_ready(ser: serial.Serial, timeout: float = 30.0) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout:
        linea = ser.readline().decode("utf-8", errors="ignore").strip()
        if linea == "READY":
            return True
        if linea.startswith("#") and linea:
            console.print(f"[dim yellow][firmware][/] {linea}")
    return False


def _parsear_trama(linea: str):
    if not linea.startswith(PROTOCOLO_PREFIJO):
        return None
    partes = linea[len(PROTOCOLO_PREFIJO):].split(",")
    if len(partes) != len(NOMBRES_CANALES):
        return None
    try:
        return [float(p) for p in partes]
    except ValueError:
        return None


def _parsear_numero(texto: str):
    texto = texto.strip()
    if not PATRON_NUMERO.match(texto):
        return None
    return float(texto)


def capturar_angulos(ser: serial.Serial, angulo_codo: float, angulo_muneca: float,
                      duracion_s: float, capturador: CapturadorVentanas,
                      sesion_id: str) -> list:
    """Lee tramas crudas durante duracion_s segundos, filtra y ventanea
    en tiempo real usando una barra de progreso Rich."""
    registros = []
    t0 = time.time()
    n_canales = len(NOMBRES_CANALES)

    # Reemplazamos el contador por un componente de barra horizontal fluido
    with Progress(
        TextColumn("[bold yellow]└─►[/] [cyan]{task.description:<22}[/]"),
        BarColumn(bar_width=30, style="dim", complete_style="yellow"),
        MofNCompleteColumn(),
        console=console
    ) as progress:
        
        # Estimación en base a ≈50Hz (Paso_ms) para inicializar el total de la barra
        total_estimado = int(duracion_s * 50)
        tarea = progress.add_task("Muestreando músculos...", total=total_estimado)

        while time.time() - t0 < duracion_s:
            raw = ser.readline()
            if not raw:
                continue
            linea = raw.decode("utf-8", errors="ignore").strip()

            valores = _parsear_trama(linea)
            if valores is not None:
                vector_features = capturador.procesar_trama(valores)
                if vector_features is not None:
                    ts_iso = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                    registros.append([sesion_id, ts_iso] + vector_features + [angulo_codo, angulo_muneca])
                    
                    # Forzamos que avance la barra visual de forma proporcional
                    progreso_actual = int((time.time() - t0) * 50)
                    progress.update(tarea, completed=min(progreso_actual, total_estimado))

        progress.update(tarea, completed=total_estimado)

    return registros


def leer_angulos() -> tuple:
    """Solicita ambos ángulos objetivo usando el sistema de Prompts validantes de Rich."""
    while True:
        console.print("\n[bold yellow]» Configuración de Ángulo Target[/]")
        entrada = Prompt.ask(
            "  Ángulos a capturar [cyan]codo,muneca[/] (ej: [dim]90,0[/]), [dim]0,0[/] reposo, o [bold red]q[/] para terminar"
        ).strip()
        
        if entrada.lower() == "q":
            return None, None

        partes = [p.strip() for p in entrada.split(",")]
        if len(partes) != 2:
            console.print("[bold red]✗[/] Formato esperado: <angulo_codo>,<angulo_muneca> (ej: '90,0')")
            continue

        codo = _parsear_numero(partes[0])
        muñeca = _parsear_numero(partes[1])

        if codo is None or muñeca is None:
            console.print("[bold red]✗[/] Entrada no válida. No se permiten letras, 'nan', 'inf', ni notación científica.")
            continue

        if not (ANGULO_MIN <= codo <= ANGULO_MAX):
            console.print(f"[bold red]✗[/] angulo_codo fuera de los límites [{ANGULO_MIN}, {ANGULO_MAX}].")
            continue
        if not (ANGULO_MIN <= muñeca <= ANGULO_MAX):
            console.print(f"[bold red]✗[/] angulo_muneca fuera de los límites [{ANGULO_MIN}, {ANGULO_MAX}].")
            continue

        return codo, muñeca


def resumen_csv(path: str):
    """Muestra distribución de combinaciones de ángulos registradas en una Tabla estructurada."""
    if not os.path.exists(path):
        return
    combinaciones = {}
    sesiones_vistas = set()
    with open(path, "r") as f:
        reader = csv.DictReader(f)
        for fila in reader:
            try:
                key = (float(fila[COL_ANGULO_CODO]), float(fila[COL_ANGULO_MUNECA]))
                combinaciones[key] = combinaciones.get(key, 0) + 1
                if "sesion_id" in fila:
                    sesiones_vistas.add(fila["sesion_id"])
            except (ValueError, KeyError):
                pass

    if sesiones_vistas:
        console.print(f"  [dim]Sesiones distintas acumuladas en el CSV: "
                      f"[bold cyan]{len(sesiones_vistas)}[/][/]")

    if combinaciones:
        tabla = Table(title="\nDistribución de Posiciones en el Dataset", box=box.ROUNDED, border_style="dim")
        tabla.add_column("Codo Target", justify="right", style="cyan")
        tabla.add_column("Muñeca Target", justify="right", style="cyan")
        tabla.add_column("Vectores Registrados", justify="right", style="bold green")
        
        for (codo, muñeca) in sorted(combinaciones):
            tabla.add_row(f"{codo:6.1f}°", f"{muñeca:6.1f}°", f"{combinaciones[(codo, muñeca)]:4d}")
            
        console.print(tabla)


# ------------------------------------------------------------------------------
def ejecutar_captura_interactiva(ser: serial.Serial, duracion_s: int = DURACION_CAPTURA_S,
                                  ruta_salida: str = DATA_PATH,
                                  ruta_sesiones: str = RUTA_SESIONES) -> int:
    os.makedirs(os.path.dirname(os.path.abspath(ruta_salida)), exist_ok=True)
    archivo_nuevo = not os.path.exists(ruta_salida)

    capturador = CapturadorVentanas()
    total = 0

    sesion_id = generar_sesion_id()
    t_inicio_sesion = datetime.now(timezone.utc)
    repeticiones_por_posicion = {}   # {"codo,muneca": n_vectores}

    console.print(f"  [dim]Sesión de captura:[/] [bold cyan]{sesion_id}[/]")

    with open(ruta_salida, "a", newline="") as f:
        writer = csv.writer(f)
        if archivo_nuevo:
            writer.writerow(COLUMNAS_CSV)

        console.print(Panel(
            "[bold green]Sistema Listo.[/] Posiciona el brazo en los ángulos deseados\n"
            "y mantén la contracción/postura fija de forma estable durante la captura.",
            border_style="green"
        ))

        while True:
            angulo_codo, angulo_muneca = leer_angulos()
            if angulo_codo is None:
                break

            console.print(f"\n  Prepárate para sostener: [bold cyan]Codo {angulo_codo:.0f}° | Muñeca {angulo_muneca:.0f}°[/] durante {duracion_s}s...")
            for s in range(3, 0, -1):
                console.print(f"  [bold yellow]{s}...[/]", end="\r")
                time.sleep(1.0)
            console.print(f"  [bold green]¡CAPTURANDO![/]                          ")

            capturador.reset()  
            registros = capturar_angulos(ser, angulo_codo, angulo_muneca,
                                          duracion_s, capturador, sesion_id)

            for reg in registros:
                # sesion_id (str) y timestamp (str) van tal cual; el resto son floats
                writer.writerow(reg[:2] + [f"{v:.6f}" for v in reg[2:]])
            f.flush()

            total += len(registros)
            clave_pos = f"{angulo_codo:.1f},{angulo_muneca:.1f}"
            repeticiones_por_posicion[clave_pos] = repeticiones_por_posicion.get(clave_pos, 0) + 1

            console.print(f"  [bold green]✓[/] {len(registros)} vectores guardados. [dim](Total acumulado en sesión: {total})[/]")

            continuar = Prompt.ask(
                "\n  ¿Capturar otra combinación?", 
                choices=["s", "q"], 
                default="s"
            )
            if continuar == "q":
                break

    t_fin_sesion = datetime.now(timezone.utc)
    duracion_real_s = (t_fin_sesion - t_inicio_sesion).total_seconds()

    # --- Persistir el registro de esta sesión (Tabla 6.6) -------------------
    registrar_sesion({
        "sesion_id": sesion_id,
        "fecha_inicio": t_inicio_sesion.isoformat(timespec="seconds"),
        "fecha_fin": t_fin_sesion.isoformat(timespec="seconds"),
        "duracion_total_s": round(duracion_real_s, 1),
        "duracion_por_captura_s": duracion_s,
        "num_posiciones_angulares": len(repeticiones_por_posicion),
        "repeticiones_por_posicion": repeticiones_por_posicion,
        "total_repeticiones": sum(repeticiones_por_posicion.values()),
        "total_vectores": total,
        "dataset_csv": os.path.abspath(ruta_salida),
    }, ruta=ruta_sesiones)

    console.print(f"\n  Dataset: [dim]{os.path.abspath(ruta_salida)}[/]")
    console.print(f"  Registro de sesión guardado en: [dim]{os.path.abspath(ruta_sesiones)}[/]")
    console.print(f"  Vectores añadidos en esta sesión: [bold green]{total}[/]")
    resumen_csv(ruta_salida)
    return total


# ------------------------------------------------------------------------------
def main():
    """Punto de entrada independiente con estética de cabecera unificada."""
    parser = argparse.ArgumentParser(description="Captura dataset EMG — DSP en Python")
    parser.add_argument("--port", default=PORT)
    parser.add_argument("--duracion", type=int, default=DURACION_CAPTURA_S,
                        help="Segundos de captura por combinación de ángulos")
    args = parser.parse_args()

    # Cabecera estructurada idéntica a main.py
    console.print("\n" + "=" * 60)
    console.print("  [bold magenta]CAPTURA DE DATASET — Extracción de Características Crudas[/]")
    console.print("=" * 60)
    
    info_panel = (
        f"Puerto Serial: [cyan]{args.port}[/]\n"
        f"Duración: [cyan]{args.duracion} segundos[/] por muestra\n"
        f"Canales activos: [cyan]{NOMBRES_CANALES}[/]\n\n"
        f"[dim]Nota: Las características se guardan en crudo. La estandarización\n"
        f"en %MVC se procesará de manera offline en la siguiente etapa.[/]"
    )
    console.print(Panel(info_panel, title="[bold cyan]Configuración de Captura[/]", expand=False))

    try:
        ser = serial.Serial(args.port, BAUDRATE, timeout=2.0)
    except serial.SerialException as e:
        console.print(f"[bold red]✗ Error al abrir puerto:[/] {e}")
        sys.exit(1)

    time.sleep(2.0)   

    console.print("[cyan]Esperando READY del firmware...[/]")
    if not esperar_ready(ser):
        console.print("[bold red]✗ READY no recibido.[/] Verifica el firmware.")
        ser.close()
        sys.exit(1)

    ser.reset_input_buffer()   

    ejecutar_captura_interactiva(ser, args.duracion, DATA_PATH)
    ser.close()


if __name__ == "__main__":
    main()
