# =============================================================================
# estandarizacion.py — Normalización %MVC offline de un dataset ya capturado
# =============================================================================
# Ubicación prevista: src/processing/estandarizacion.py (o data/estandarizacion.py
# si se prefiere ejecutar como script suelto, junto a captura.py)
#
# Responsabilidad ÚNICA de este módulo: tomar un CSV de features CRUDOS
# (generado por data/captura.py) + una calibración guardada (baseline/MVC,
# generada por calibracion.py -> CalibradorEMG.guardar()) y producir un
# CSV normalizado en %MVC, listo para entrenar el regresor.
#
# Este módulo NO abre puerto serial ni depende de hardware — trabaja
# exclusivamente sobre archivos. Por eso vive separado de calibracion.py
# (que sí opera en vivo). Ver la discusión de diseño: se decidió guardar
# los datasets en CRUDO (no normalizados en el momento de la captura) para
# poder recalcular la normalización sin tener que recapturar datos si se
# corrige un bug de calibración o se ajusta el criterio de clipping.
#
# SUPUESTO DE DISEÑO — una sola calibración por archivo de dataset:
#   El proyecto asume que cada CSV de datos crudos (datos_emg.csv)
#   corresponde a UNA sola sesión de calibración (ver nota en
#   calibracion.py: "el diseño del proyecto asume calibración por
#   sesión, no reutilización entre sesiones"). Si en el futuro se
#   capturan datos de múltiples sesiones en un mismo archivo, este
#   script debe extenderse para leer una columna "sesion_id" por fila
#   y aplicar la calibración correspondiente a cada una (actualmente
#   NO implementado — ver TODO al final del archivo).
#
# Normalización aplicada, feature por feature:
#   %MVC = (valor_crudo - baseline) / (mvc - baseline) * 100
#
# El resultado se guarda SIN perder trazabilidad:
#   - columnas pct_mvc_<feature>       -> normalizado y clipeado [0,100],
#                                          esto es lo que debe usar el
#                                          entrenamiento del modelo.
#   - columnas pct_mvc_<feature>_crudo -> normalizado SIN clip, para
#                                          auditar saturaciones (valores
#                                          >100% indican que la contracción
#                                          real superó el MVC calibrado).
#   - columna saturado_algun_canal     -> True si alguna feature de esa
#                                          fila superó el 100% del MVC.
#
# Uso:
#   python src/processing/estandarizacion.py
#   python src/processing/estandarizacion.py --datos data/datos_emg.csv \
#       --calibracion data/calibracion.json --salida data/datos_emg_normalizado.csv
# =============================================================================

# =============================================================================
# estandarizacion.py — Normalización %MVC offline de un dataset (Visual Rich)
# =============================================================================

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from emg_arm.config import NOMBRES_FEATURES, DATA_PATH, RUTA_CALIBRACION_DEFAULT

# Importaciones de la estética unificada Rich
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

RUTA_SALIDA_DEFAULT = os.path.join(
    os.path.dirname(__file__), "..", "..", "data", "datos_emg_normalizado.csv"
)

UMBRAL_ALERTA_SATURACION_PCT = 5.0  


def cargar_calibracion(ruta: str) -> dict:
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No existe archivo de calibración: {ruta}")

    with open(ruta, "r") as f:
        data = json.load(f)

    if data.get("features") != NOMBRES_FEATURES:
        raise ValueError(
            "El archivo de calibración fue guardado con un orden de "
            "features distinto al actual en config.py. Recalibrar antes "
            "de normalizar, o alinear NOMBRES_FEATURES manualmente."
        )

    baseline = np.array(data["baseline"], dtype=np.float64)
    mvc = np.array(data["mvc"], dtype=np.float64)
    return {"baseline": baseline, "mvc": mvc}


def normalizar_dataframe(df: pd.DataFrame, calibracion: dict) -> pd.DataFrame:
    baseline = calibracion["baseline"]
    mvc = calibracion["mvc"]
    rango = mvc - baseline

    if np.any(rango <= 1e-6):
        raise ValueError(
            "La calibración tiene rango (mvc - baseline) <= 0 en alguna "
            "feature. No se puede normalizar de forma confiable — "
            "recalibrar esa sesión."
        )

    df = df.copy()
    columnas_saturacion = []

    for i, feat in enumerate(NOMBRES_FEATURES):
        if feat not in df.columns:
            raise KeyError(
                f"La columna '{feat}' no está en el CSV de datos. "
                f"¿El CSV fue generado con el mismo config.py actual?"
            )

        crudo = df[feat].to_numpy(dtype=np.float64)
        pct_sin_clip = (crudo - baseline[i]) / rango[i] * 100.0
        pct_clipeado = np.clip(pct_sin_clip, 0.0, 100.0)

        df[f"pct_mvc_{feat}"] = pct_clipeado
        df[f"pct_mvc_{feat}_crudo"] = pct_sin_clip

        col_sat = f"_saturado_{feat}"
        df[col_sat] = pct_sin_clip > 100.0
        columnas_saturacion.append(col_sat)

    df["saturado_algun_canal"] = df[columnas_saturacion].any(axis=1)
    df.drop(columns=columnas_saturacion, inplace=True)

    return df


def reportar_saturaciones(df: pd.DataFrame):
    """Imprime un resumen estructurado y visual de las filas y features que

    saturaron el MVC calibrado utilizando componentes de Rich."""
    total = len(df)
    if total == 0:
        console.print("[bold yellow]⚠ [estandarización] El dataset está vacío, nada que reportar.[/]")
        return

    n_filas_saturadas = int(df["saturado_algun_canal"].sum())
    pct_filas = 100 * n_filas_saturadas / total

    # Resumen general del estado de saturación
    console.print(f"\n[bold cyan]📊 Análisis de Calidad del Dataset[/]")
    console.print(f"  Filas con alguna saturación: [bold yellow]{n_filas_saturadas}[/]/[dim]{total}[/] ([bold yellow]{pct_filas:.1f}%[/])")

    # Tabla estructurada para el detalle por feature
    tabla = Table(title="\nDetalle de Saturaciones por Característica (Valores > 100% MVC)", box=box.ROUNDED, border_style="dim")
    tabla.add_column("Característica (Feature)", style="cyan")
    tabla.add_column("Filas Saturadas", justify="right", style="bold yellow")
    tabla.add_column("Máximo Alcanzado", justify="right", style="magenta")

    tiene_saturaciones = False
    for feat in NOMBRES_FEATURES:
        col = f"pct_mvc_{feat}_crudo"
        n_sat = int((df[col] > 100.0).sum())
        if n_sat > 0:
            tiene_saturaciones = True
            maximo = df[col].max()
            tabla.add_row(feat, f"{n_sat:d}", f"{maximo:.1f}%")

    if tiene_saturaciones:
        console.print(tabla)
    else:
        console.print("  [bold green]✓[/] Ninguna característica individual presentó saturación de MVC.")

    # Alertas basadas en el umbral definido
    if pct_filas > UMBRAL_ALERTA_SATURACION_PCT:
        alerta_msg = (
            f"[bold red]ALERTA:[/] El [bold]{pct_filas:.1f}%[/] de las filas superan el umbral de tolerancia ({UMBRAL_ALERTA_SATURACION_PCT}%).\n\n"
            f"[dim]Esto sugiere fuertemente que el MVC calibrado en vivo quedó por debajo del esfuerzo real\n"
            f"ejecutado durante la captura. Para evitar la pérdida de dinámica por el recorte (clip a 100%),\n"
            f"se recomienda recalibrar aplicando mayor fuerza máxima en las contracciones voluntarias.[/]"
        )
        console.print("\n", Panel(alerta_msg, title="[bold red]Saturación Excesiva[/]", border_style="red", expand=False))
    else:
        console.print(f"\n  [bold green]✓[/] Nivel de saturación óptimo (<{UMBRAL_ALERTA_SATURACION_PCT}%). "
                      f"[dim]Los picos aislados son normales y el clipeado adaptativo es suficiente.[/]")


def main():
    parser = argparse.ArgumentParser(
        description="Normaliza a %MVC un dataset EMG crudo, usando una calibración guardada."
    )
    parser.add_argument("--datos", default=DATA_PATH,
                         help="CSV de features crudos (salida de captura.py)")
    parser.add_argument("--calibracion", default=RUTA_CALIBRACION_DEFAULT,
                         help="JSON de calibración (salida de CalibradorEMG.guardar())")
    parser.add_argument("--salida", default=RUTA_SALIDA_DEFAULT,
                         help="Ruta del CSV normalizado de salida")
    args = parser.parse_args()

    # Cabecera estructurada unificada con el resto de la suite
    console.print("\n" + "=" * 60)
    console.print("  [bold magenta]ESTANDARIZACIÓN %MVC — Normalización Offline de Dataset[/]")
    console.print("=" * 60)
    
    config_info = (
        f"Datos crudos: [cyan]{args.datos}[/]\n"
        f"Calibración:  [cyan]{args.calibracion}[/]\n"
        f"Salida CSV:   [cyan]{args.salida}[/]"
    )
    console.print(Panel(config_info, title="[bold cyan]Rutas de Trabajo[/]", expand=False))

    if not os.path.exists(args.datos):
        console.print(f"[bold red]✗ Error:[/] No existe el archivo de datos crudos en: [dim]{args.datos}[/]")
        sys.exit(1)

    df = pd.read_csv(args.datos)
    console.print(f"\n[cyan]ℹ[/] Se cargaron [bold green]{len(df)}[/] filas desde el dataset crudo.")

    try:
        calibracion = cargar_calibracion(args.calibracion)
        df_normalizado = normalizar_dataframe(df, calibracion)
    except Exception as e:
        console.print(f"[bold red]✗ Error de normalización:[/] {e}")
        sys.exit(1)

    # Llama al reporte visual formateado con Rich
    reportar_saturaciones(df_normalizado)

    os.makedirs(os.path.dirname(os.path.abspath(args.salida)), exist_ok=True)
    df_normalizado.to_csv(args.salida, index=False)
    console.print(f"\n[bold green]✓[/] Dataset normalizado exportado correctamente en:")
    console.print(f"  [dim]└─► {os.path.abspath(args.salida)}[/]\n")


if __name__ == "__main__":
    main()

# =============================================================================
# TODO — soporte multi-sesión (no implementado):
#   Si en algún momento se capturan datos de varias sesiones de calibración
#   distintas en un mismo datos_emg.csv, agregar:
#     1. Columna "sesion_id" en cada fila (escrita por captura.py).
#     2. Un calibracion.json indexado por sesion_id, en vez de un único
#        baseline/mvc global (ver ejemplo de estructura discutido en el
#        diseño: {"<sesion_id>": {"baseline": [...], "mvc": [...]}, ...}).
#     3. normalizar_dataframe() debería agrupar por sesion_id y aplicar
#        la calibración correspondiente a cada grupo, en vez de una única
#        calibración global para todo el DataFrame.
# =============================================================================