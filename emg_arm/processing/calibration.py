# =============================================================================
# calibracion.py — Calibración baseline/MVC por canal y por feature
# =============================================================================
# Ubicación prevista: src/processing/calibracion.py
#
# Responsabilidad ÚNICA de este módulo: ejecutar el protocolo de
# calibración (4 fases) sobre datos en vivo desde el Arduino, y aplicar
# esa calibración a vectores de features individuales (normalizar()).
#
# La normalización de un DATASET YA CAPTURADO (offline, sobre un CSV
# completo) se hace en estandarizacion.py, no aquí — separación de
# responsabilidades: este módulo trabaja en vivo, con serial abierto;
# estandarizacion.py trabaja sobre archivos, sin hardware.
#
# Protocolo de calibración (4 fases, sobre las tramas crudas "S,..." que
# transmite el firmware v4, reutilizando el mismo CapturadorVentanas que
# usan captura.py y serial_bridge.py):
#   Fase 1 — REPOSO (los 3 músculos relajados): baseline de las 12 features.
#   Fase 2 — MVC BÍCEPS (contraer solo bíceps): MVC de features[0:4].
#   Fase 3 — MVC TRÍCEPS (contraer solo tríceps): MVC de features[4:8].
#   Fase 4 — MVC ANTEBRAZO (contraer solo pronator teres): MVC de
#            features[8:12].
#
# Normalización aplicada, feature por feature (12 pares baseline/MVC
# independientes, no solo sobre RMS):
#   %MVC = clip((valor - baseline) / (MVC - baseline) * 100, 0, 100)
#
# ADVERTENCIA DE DISEÑO — sin validar experimentalmente:
#   Normalizar ZCR (cruces por cero) con la misma fórmula que RMS/MAV/WL
#   asume que ZCR también crece de forma monótona con la intensidad de
#   contracción. Esta suposición no está confirmada en este proyecto — es
#   una extensión del criterio original (que solo normalizaba RMS), no un
#   resultado medido. Si la validación experimental (Capítulo 8) muestra
#   que ZCR no se comporta así, debe excluirse de esta normalización y
#   dejarse crudo en el vector final que recibe el regresor.
#
# NOTA DE DUPLICACIÓN:
#   PROTOCOLO_PREFIJO ("S,") y la lógica de parseo de tramas están
#   duplicadas aquí, en data/captura.py y en src/core/serial_bridge.py.
#   No se unificó en un único punto de definición. Si se cambia el
#   protocolo serial, actualizar los 3 archivos.
# =============================================================================

# =============================================================================
# calibracion.py — Calibración baseline/MVC por canal y por feature (Rich)
# =============================================================================

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from emg_arm.config import NOMBRES_FEATURES, NOMBRES_CANALES, N_FEATURES_POR_CANAL
from emg_arm.processing.dsp import CapturadorVentanas

# Importaciones de la estética unificada Rich
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, TextColumn, BarColumn, MofNCompleteColumn
from rich import box

console = Console()

PROTOCOLO_PREFIJO = "S,"

RUTA_CALIBRACION_DEFAULT = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "calibracion.json"
)


def _parsear_trama(linea: str, n_canales: int):
    if not linea.startswith(PROTOCOLO_PREFIJO):
        return None
    partes = linea[len(PROTOCOLO_PREFIJO):].split(",")
    if len(partes) != n_canales:
        return None
    try:
        return [float(p) for p in partes]
    except ValueError:
        return None


def _leer_vectores(ser, capturador: CapturadorVentanas, duracion_s: float,
                    etiqueta: str = "") -> list:
    """Lee tramas crudas desde `ser` durante duracion_s segundos usando una
    barra de progreso dinámica de Rich."""
    registros = []
    t0 = time.time()
    n_canales = len(NOMBRES_CANALES)

    with Progress(
        TextColumn("[bold yellow]└─►[/] [cyan]{task.description:<25}[/]"),
        BarColumn(bar_width=30, style="dim", complete_style="magenta"),
        MofNCompleteColumn(),
        console=console
    ) as progress:
        
        tarea = progress.add_task(f"{etiqueta}...", total=int(duracion_s * 50))
        
        while time.time() - t0 < duracion_s:
            raw = ser.readline()
            if not raw:
                continue
            linea = raw.decode("utf-8", errors="ignore").strip()

            if linea.startswith("#"):
                continue

            valores = _parsear_trama(linea, n_canales)
            if valores is None:
                continue

            vector = capturador.procesar_trama(valores)
            if vector is not None:
                registros.append(vector)
                progreso_actual = int((time.time() - t0) * 50)
                progress.update(tarea, completed=min(progreso_actual, int(duracion_s * 50)))

        progress.update(tarea, completed=int(duracion_s * 50))

    return registros


class CalibradorEMG:
    """Gestiona la calibración baseline/MVC por canal y por feature, y la
    normalización %MVC de vectores de features en vivo."""

    def __init__(self):
        self.baseline = None
        self.mvc = None
        self.calibrado = False
        self._advertido_sin_calibrar = False

    def ejecutar(self, ser, capturador: CapturadorVentanas,
                 duracion_reposo_s: float = 3.0,
                 duracion_mvc_s: float = 4.0):  # Ajustado por defecto a 4s (rango 3-5s)
        """Ejecuta el protocolo avanzado anti-fatiga con 3 intentos de MVC por músculo,
        descansos de 45s entre intentos y descansos largos de 2.5 min entre músculos."""
        n_feat = len(NOMBRES_FEATURES)
        baseline = np.zeros(n_feat)
        mvc = np.full(n_feat, 1.0) 

        console.print("\n" + "=" * 65)
        console.print("  [bold magenta]CALIBRACIÓN sEMG AVANZADA — Protocolo Clínico de Normalización[/]")
        console.print("=" * 65)

        # --- FASE 1: REPOSO GENERAL ---
        capturador.reset()
        console.print("\n[bold cyan][calibración] FASE 1 — REPOSO GENERAL[/]")
        console.print("  [dim]Relaja completamente todo el brazo. Respira hondo y suelta la tensión.[/]")
        self._contar_regresivo(segundos=3, mensaje_final="[bold blue]¡MANTÉN EL REPOSO Y QUÉDATE QUIETO![/]")
        vectores = _leer_vectores(ser, capturador, duracion_reposo_s, etiqueta="Capturando Reposo")
        if vectores:
            baseline = np.mean(np.array(vectores), axis=0)
        else:
            console.print("[bold red]⚠ ADVERTENCIA:[/] Sin vectores válidos en reposo. Baseline forzado a 0.")

        # --- FASES DE MVC: ITERACIÓN DE MÚSCULOS ---
        for i, nombre_canal in enumerate(NOMBRES_CANALES):
            idx_lo = i * N_FEATURES_POR_CANAL
            idx_hi = idx_lo + N_FEATURES_POR_CANAL

            console.print(f"\n[bold magenta]=======================================================[/]")
            console.print(f"  [bold yellow]MÚSCULO OBJETIVO:[/] [bold green]{nombre_canal.upper()}[/]")
            console.print(f"  [dim]Preparación anatómica: Coloca el brazo en la posición articular recomendada.[/]")
            console.print(f"  [dim]Realiza 1 o 2 contracciones submáximas de práctica antes de continuar.[/]")
            console.print(f"[bold magenta]=======================================================[/]")
            
            input("\n  [Presiona ENTER cuando estés listo para iniciar los 3 intentos de MVC]...")

            # Lista para almacenar el máximo absoluto alcanzado por cada una de las 4 features en cada intento
            maximos_intentos = []

            # 3 Intentos de MVC por músculo
            for intento in range(1, 4):
                console.print(f"\n  [bold cyan]» INTENTO {intento}/3 — MVC {nombre_canal.upper()}[/]")
                console.print(f"    ¡Contrae [bold red]SOLO[/] el '[bold green]{nombre_canal}[/]' con fuerza máxima explosiva!")
                
                # POLIMORFISMO: Aquí sí le exigimos la contracción destructiva
                self._contar_regresivo(3, mensaje_final="[bold red]¡CONTRAE A MÁXIMA FUERZA YA![/]")
                
                capturador.reset()
                vectores_intento = _leer_vectores(
                    ser, capturador, duracion_mvc_s, etiqueta=f"Grabando Intento {intento}"
                )
                
                if vectores_intento:
                    arr_intento = np.array(vectores_intento)
                    # Extraer el pico máximo de cada una de las 4 características de este canal en este intento
                    picos_intento = np.max(arr_intento[:, idx_lo:idx_hi], axis=0)
                    maximos_intentos.append(picos_intento)
                    console.print(f"    [bold green]✓ Intento {intento} registrado.[/] Pico RMS detectado: {picos_intento[0]:.2f}")
                else:
                    console.print(f"    [bold red]✗[/] Intento fallido sin datos.")

                # Descanso entre intentos (Mismo músculo): 30-60 segundos (Fijado en 45s)
                if intento < 3:
                    self._ejecutar_pausa_visual(
                        segundos=45, 
                        mensaje=f"[yellow]Descanso intra-músculo (Intento {intento} → {intento+1})[/]"
                    )

            # Procesar los intentos: registrar el mayor valor absoluto entre los 3 intentos para normalizar sEMG
            if maximos_intentos:
                mvc[idx_lo:idx_hi] = np.max(np.array(maximos_intentos), axis=0)
            else:
                console.print(f"  [bold red]⚠ CRÍTICO:[/] Sin datos en ningún intento de '{nombre_canal}'. Rango por defecto.")

            # Descanso largo entre diferentes músculos (Bíceps → Tríceps → Antebrazo): 2-3 minutos (Fijado en 120s por eficiencia práctica)
            if i < len(NOMBRES_CANALES) - 1:
                self._ejecutar_pausa_visual(
                    segundos=120, 
                    mensaje=f"[bold green]✓ Canal {nombre_canal.upper()} cerrado.[/] [bold yellow]Pausa de recuperación metabólica obligatoria[/]"
                )

        # Validación matemática de seguridad anti división por cero
        rango = mvc - baseline
        invalidos = rango <= 1e-6
        if np.any(invalidos):
            nombres_invalidos = [NOMBRES_FEATURES[idx] for idx in range(n_feat) if invalidos[idx]]
            console.print(Panel(
                f"[bold red]Rango baseline-MVC inválido en:[/] {nombres_invalidos}\n"
                f"[dim]Se fuerza un rango mínimo de 1.0 para evitar divisiones críticas por cero.[/]",
                title="[bold red]Alerta de Calibración[/]", border_style="red", expand=False
            ))
            mvc[invalidos] = baseline[invalidos] + 1.0

        self.baseline = baseline
        self.mvc = mvc
        self.calibrado = True

        # --- TABLA RESUMEN ESTILIZADA DE PICOS ENCONTRADOS ---
        tabla = Table(title="\n[bold magenta]Resumen de Calibración sEMG Final (Picos de 3 Intentos)[/]", box=box.ROUNDED, border_style="dim")
        tabla.add_column("Característica (Feature)", style="bold cyan")
        tabla.add_column("Baseline (Reposo)", justify="right", style="green")
        tabla.add_column("→", justify="center")
        tabla.add_column("MVC Absoluto Registrado", justify="right", style="bold magenta")

        for idx, nombre in enumerate(NOMBRES_FEATURES):
            tabla.add_row(nombre, f"{baseline[idx]:10.3f}", "→", f"{mvc[idx]:10.3f}")
            
        console.print(tabla)
        console.print("[bold green]✓[/] Proceso de calibración anti-fatiga completado con éxito.\n")

    @staticmethod
    def _contar_regresivo(segundos: int = 3, mensaje_final: str = ""):
            """Cuenta regresiva polimórfica que adapta su salida visual según la fase."""
            for s in range(segundos, 0, -1):
                console.print(f"  [bold yellow]{s}...[/]", end="\r")
                time.sleep(1.0)
            
            # Limpia la línea anterior e imprime el comportamiento específico de la fase
            console.print(f"  {mensaje_final}               ")

    @staticmethod
    def _ejecutar_pausa_visual(segundos: int, mensaje: str):
        """Muestra una barra de progreso Rich en cuenta regresiva para los descansos biológicos."""
        with Progress(
            TextColumn("⏳ {task.description}"),
            BarColumn(bar_width=25, style="dim", complete_style="green"),
            TextColumn("[bold clock]{task.fields[restante]}s restantes[/]"),
            console=console
        ) as prog_descanso:
            tarea = prog_descanso.add_task(mensaje, total=segundos, restante=segundos)
            for s in range(segundos):
                time.sleep(1.0)
                prog_descanso.update(tarea, advance=1, restante=segundos - s - 1)

    def normalizar(self, vector_crudo: list) -> list:
        if not self.calibrado:
            if not self._advertido_sin_calibrar:
                console.print("[bold yellow]⚠ [calibración] ADVERTENCIA:[/] normalizar() llamado sin calibración activa — Devolviendo crudos.")
                self._advertido_sin_calibrar = True
            return list(vector_crudo)

        arr = np.array(vector_crudo, dtype=np.float64)
        pct = (arr - self.baseline) / (self.mvc - self.baseline) * 100.0
        pct = np.clip(pct, 0.0, 100.0)
        return pct.tolist()

    def normalizar_con_diagnostico(self, vector_crudo: list) -> dict:
        if not self.calibrado:
            if not self._advertido_sin_calibrar:
                console.print("[bold yellow]⚠ [calibración] ADVERTENCIA:[/] normalizar_con_diagnostico() llamado sin calibración activa.")
                self._advertido_sin_calibrar = True
            n = len(vector_crudo)
            return {
                "clipeado": list(vector_crudo),
                "sin_clip": list(vector_crudo),
                "saturado": [False] * n,
            }

        arr = np.array(vector_crudo, dtype=np.float64)
        pct_sin_clip = (arr - self.baseline) / (self.mvc - self.baseline) * 100.0
        pct_clipeado = np.clip(pct_sin_clip, 0.0, 100.0)
        saturado = pct_sin_clip > 100.0

        return {
            "clipeado": pct_clipeado.tolist(),
            "sin_clip": pct_sin_clip.tolist(),
            "saturado": saturado.tolist(),
        }

    def guardar(self, ruta: str = RUTA_CALIBRACION_DEFAULT):
        if not self.calibrado:
            console.print("[bold red]✗[/] No hay calibración activa para guardar.")
            return
        os.makedirs(os.path.dirname(os.path.abspath(ruta)), exist_ok=True)
        data = {
            "features": NOMBRES_FEATURES,
            "baseline": self.baseline.tolist(),
            "mvc": self.mvc.tolist(),
        }
        with open(ruta, "w") as f:
            json.dump(data, f, indent=2)

    def cargar(self, ruta: str = RUTA_CALIBRACION_DEFAULT) -> bool:
        if not os.path.exists(ruta):
            console.print(f"[bold red]✗[/] No existe archivo de calibración en: [dim]{ruta}[/]")
            return False
        with open(ruta, "r") as f:
            data = json.load(f)
        if data.get("features") != NOMBRES_FEATURES:
            console.print("[bold red]⚠ [calibración] ERROR:[/] Desajuste en el orden o cantidad de features del archivo. No se cargará.")
            return False
        self.baseline = np.array(data["baseline"])
        self.mvc = np.array(data["mvc"])
        self.calibrado = True
        return True