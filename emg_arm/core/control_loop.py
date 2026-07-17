# =============================================================================
# control_loop.py — Loop de control en tiempo real (inferencia + servos)
# =============================================================================
# Extraído de main.py para cumplir SRP. Este módulo maneja únicamente:
#   - Hilo de control (hilo_control): lee features de la cola, predice
#     ángulos, envía comandos al Arduino.
#   - Panel de estado en vivo (Live Rich).
#   - Modo evaluación de trayectoria con ángulos objetivo conocidos.
#   - Cálculo de resumen de sesión de control (Tabla 8.2).
#
# NO contiene: menú principal, flujo de entrenamiento, UI de selección.
# Eso vive en emg_arm/core/main.py.
# =============================================================================

import os
import sys
import time
import threading
import queue

import numpy as np

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.prompt import Prompt, Confirm
from rich.live import Live
from rich import box

from emg_arm.config import (PORT, BAUDRATE, INTERVALO_CONTROL,
                         NOMBRES_FEATURES, RUTA_CALIBRACION_DEFAULT)
from emg_arm.communication.serial_bridge import SerialBridge
from emg_arm.models.predictor import EMGPredictor
from data.capture import (generar_sesion_id, registrar_sesion, leer_angulos)

console = Console()

# ---------------------------------------------------------------------------
# Constantes de control
# ---------------------------------------------------------------------------
RUTA_SESIONES_CONTROL = os.path.join(
    os.path.dirname(__file__), "data", "sesiones_control.json"
)
LOG_UMBRAL_CAMBIO = 2.0
INTERVALO_MEDICION_LATENCIA_S = 1.0

_IDX_RMS_BICEPS    = NOMBRES_FEATURES.index("rms_biceps")
_IDX_RMS_TRICEPS   = NOMBRES_FEATURES.index("rms_triceps")
_IDX_RMS_ANTEBRAZO = NOMBRES_FEATURES.index("rms_antebrazo")

# Cola compartida entre hilo_serial (bridge.leer_muestras) y hilo_control
cola_features: queue.Queue = queue.Queue(maxsize=10)
flag_activo = threading.Event()


# =============================================================================
# Hilo de control
# =============================================================================
def _conectar_y_esperar_ready(puerto: str):
    """Abre el puerto, espera el READY del firmware con un spinner, y
    limpia el buffer. Retorna el objeto Serial o None si falló."""
    import serial
    from data.capture import esperar_ready

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


def hilo_control(bridge: SerialBridge, predictor: EMGPredictor, estado: dict,
                  registro_latencias: list):
    """Loop de control que corre en su propio hilo. Consume features de
    cola_features, predice ángulos con el regresor, y los envía al Arduino."""
    angulo_codo_ant   = -1.0
    angulo_muneca_ant = -1.0
    t_ultima_medicion = 0.0

    while flag_activo.is_set():
        try:
            features = cola_features.get(timeout=1.0)
        except queue.Empty:
            continue

        t0 = time.time()

        resultado = predictor.predecir_angulos(features)
        angulo_codo   = resultado["angulo_codo"]
        angulo_muneca = resultado["angulo_muneca"]

        if t0 - t_ultima_medicion >= INTERVALO_MEDICION_LATENCIA_S:
            envio = bridge.enviar_angulos_con_medicion(angulo_codo, angulo_muneca)
            if envio["exito"]:
                registro_latencias.append({
                    "t_unix": time.time(),
                    "latencia_electronica_ms": envio["latencia_electronica_ms"],
                    "latencia_e2e_ms": envio["latencia_e2e_ms"],
                    "angulo_codo_pred": angulo_codo,
                    "angulo_muneca_pred": angulo_muneca,
                    "objetivo_codo": estado.get("objetivo_codo"),
                    "objetivo_muneca": estado.get("objetivo_muneca"),
                })
            t_ultima_medicion = t0
        else:
            bridge.enviar_angulos(angulo_codo, angulo_muneca)

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


# =============================================================================
# Panel de estado en vivo
# =============================================================================
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


# =============================================================================
# Modo evaluación de trayectoria
# =============================================================================
def _ejecutar_modo_evaluacion(estado: dict, duracion_s: int = 5):
    """Pide ángulos objetivo al usuario, los sostiene y muestra MAE en vivo."""
    console.print(Panel.fit(
        "Ingresa ángulos objetivo conocidos y sostenlos con el brazo real.\n"
        "El sistema compara la predicción del modelo contra ese objetivo.",
        title="[bold cyan]MODO EVALUACIÓN DE TRAYECTORIA[/]", border_style="cyan"
    ))

    while True:
        objetivo_codo, objetivo_muneca = leer_angulos()
        if objetivo_codo is None:
            break

        console.print(f"\n  Sostén: [bold cyan]Codo {objetivo_codo:.0f}° | "
                       f"Muñeca {objetivo_muneca:.0f}°[/] durante {duracion_s}s...")
        for s in range(3, 0, -1):
            console.print(f"  [bold yellow]{s}...[/]", end="\r")
            time.sleep(1.0)
        console.print("  [bold green]¡EVALUANDO![/]                          ")

        estado["objetivo_codo"] = objetivo_codo
        estado["objetivo_muneca"] = objetivo_muneca
        n_antes = len(estado.get("_latencias_ref", []))
        time.sleep(duracion_s)

        muestras_pos = [r for r in estado.get("_latencias_ref", [])[n_antes:]
                         if r["objetivo_codo"] == objetivo_codo]
        if muestras_pos:
            mae_c = np.mean([abs(r["angulo_codo_pred"] - objetivo_codo) for r in muestras_pos])
            mae_m = np.mean([abs(r["angulo_muneca_pred"] - objetivo_muneca) for r in muestras_pos])
            console.print(f"  [dim]→ MAE observado en esta combinación: "
                           f"Codo {mae_c:.1f}° | Muñeca {mae_m:.1f}° ({len(muestras_pos)} muestras)[/]")

        estado["objetivo_codo"] = None
        estado["objetivo_muneca"] = None

        continuar = Prompt.ask("\n  ¿Evaluar otra combinación?", choices=["s", "q"], default="s")
        if continuar == "q":
            break


# =============================================================================
# Resumen de sesión de control (Tabla 8.2)
# =============================================================================
def _calcular_resumen_sesion_control(registro_latencias: list) -> dict:
    """Agrega los registros crudos y devuelve latencia + MAE/R² por DOF."""
    resumen = {"n_mediciones_latencia": len(registro_latencias)}

    lat_elec = [r["latencia_electronica_ms"] for r in registro_latencias]
    lat_e2e  = [r["latencia_e2e_ms"] for r in registro_latencias]
    if lat_elec:
        resumen["latencia_electronica_ms"] = {
            "promedio": float(np.mean(lat_elec)), "desv_est": float(np.std(lat_elec))
        }
        resumen["latencia_e2e_ms"] = {
            "promedio": float(np.mean(lat_e2e)), "desv_est": float(np.std(lat_e2e))
        }

    con_objetivo = [r for r in registro_latencias if r.get("objetivo_codo") is not None]
    resumen["modo"] = "evaluacion" if con_objetivo else "libre"
    if con_objetivo:
        obj_c = np.array([r["objetivo_codo"] for r in con_objetivo])
        pred_c = np.array([r["angulo_codo_pred"] for r in con_objetivo])
        obj_m = np.array([r["objetivo_muneca"] for r in con_objetivo])
        pred_m = np.array([r["angulo_muneca_pred"] for r in con_objetivo])

        def _mae_r2(y_true, y_pred):
            mae = float(np.mean(np.abs(y_true - y_pred)))
            ss_res = np.sum((y_true - y_pred) ** 2)
            ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
            r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0
            return mae, r2

        mae_codo, r2_codo = _mae_r2(obj_c, pred_c)
        mae_muneca, r2_muneca = _mae_r2(obj_m, pred_m)
        resumen["trayectoria"] = {
            "n_muestras_con_objetivo": len(con_objetivo),
            "mae_codo": mae_codo, "r2_codo": r2_codo,
            "mae_muneca": mae_muneca, "r2_muneca": r2_muneca,
        }

    return resumen


# =============================================================================
# Flujo completo "USAR" (ahora vive aquí, no en main.py)
# =============================================================================
def flujo_usar():
    """Carga el modelo y la calibración ya entrenados, conecta el Arduino,
    y corre el loop de inferencia en tiempo real."""
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

    modo_evaluacion = Confirm.ask(
        "\n  ¿Ejecutar en modo evaluación de trayectoria (ángulos objetivo "
        "conocidos, para la Tabla 8.2)? Si respondes 'No' corre en control "
        "libre (solo se registra latencia, sin MAE/R²).", default=False
    )

    sesion_id = generar_sesion_id()
    console.print(f"\n  [dim]Sesión de control:[/] [bold cyan]{sesion_id}[/]")

    estado = {"objetivo_codo": None, "objetivo_muneca": None, "_latencias_ref": None}
    registro_latencias: list = []
    estado["_latencias_ref"] = registro_latencias
    flag_activo.set()
    t_inicio_sesion = time.time()

    t_serial = threading.Thread(
        target=bridge.leer_muestras,
        args=(cola_features, flag_activo),
        daemon=True,
        name="t_serial"
    )
    t_ctrl = threading.Thread(
        target=hilo_control,
        args=(bridge, predictor, estado, registro_latencias),
        daemon=True,
        name="t_control"
    )

    t_serial.start()
    t_ctrl.start()

    console.print()
    try:
        if modo_evaluacion:
            _ejecutar_modo_evaluacion(estado)
        else:
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
    duracion_total_s = time.time() - t_inicio_sesion

    # --- Persistir la sesión de control ---------------------------------------
    resumen = _calcular_resumen_sesion_control(registro_latencias)
    resumen.update({
        "sesion_id": sesion_id,
        "fecha_inicio": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(t_inicio_sesion)),
        "duracion_total_s": round(duracion_total_s, 1),
        "puerto": puerto,
    })
    registrar_sesion(resumen, ruta=RUTA_SESIONES_CONTROL)
    console.print(f"\n[bold green]✓[/] Sesión de control guardada en: "
                  f"[dim]{os.path.abspath(RUTA_SESIONES_CONTROL)}[/]")

    if "latencia_e2e_ms" in resumen:
        console.print(f"  Latencia E2E: [bold]{resumen['latencia_e2e_ms']['promedio']:.1f} ms[/] "
                       f"(±{resumen['latencia_e2e_ms']['desv_est']:.1f} ms), "
                       f"n={resumen['n_mediciones_latencia']}")
    if "trayectoria" in resumen:
        t = resumen["trayectoria"]
        console.print(f"  MAE Codo: [bold]{t['mae_codo']:.2f}°[/] (R²={t['r2_codo']:.3f})  "
                       f"MAE Muñeca: [bold]{t['mae_muneca']:.2f}°[/] (R²={t['r2_muneca']:.3f})")

    console.print("[bold green]✓ Detenido.[/]")
