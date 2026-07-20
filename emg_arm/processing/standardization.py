# =============================================================================
# estandarizacion.py — Normalización %MVC offline de un dataset ya capturado
# =============================================================================
# Ubicación prevista: emg_arm/processing/estandarizacion.py (o data/estandarizacion.py
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
# SOPORTE MULTI-SESIÓN:
#   Cada fila del CSV crudo trae su propio "sesion_id" (escrito por
#   data/captura.py). La calibración correspondiente a cada sesión se
#   guarda indexada en el mismo calibracion.json (ver
#   CalibradorEMG.guardar(ruta, sesion_id=...)). normalizar_dataframe_
#   multisesion() agrupa el DataFrame por sesion_id y aplica a cada grupo
#   su propia calibración; si una sesión no tiene calibración indexada
#   (p. ej. filas "legacy" migradas desde el esquema sin sesion_id por
#   data/migrate.py), usa la calibración "default" (la más reciente
#   guardada) como respaldo. normalizar_dataframe() (una sola calibración
#   para todo el DataFrame) se mantiene para el caso de un dataset de una
#   única sesión, o para uso directo/pruebas.
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
#   python emg_arm/processing/estandarizacion.py
#   python emg_arm/processing/estandarizacion.py --datos data/datos_emg.csv \
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


def cargar_calibraciones(ruta: str) -> dict:
    """Carga el archivo de calibración completo, indexado por sesión.

    Retorna {"default": {"baseline":..., "mvc":...} | None,
             "sesiones": {"<sesion_id>": {"baseline":..., "mvc":...}, ...}}.

    Reconoce tanto el formato nuevo (multi-sesión, con claves "default" y
    "sesiones") como el formato plano previo ({"baseline":..., "mvc":...}
    directamente en la raíz, sin "sesiones") — en ese caso se expone como
    único "default", sin sesiones indexadas."""
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

    def _a_arrays(entrada):
        return {
            "baseline": np.array(entrada["baseline"], dtype=np.float64),
            "mvc": np.array(entrada["mvc"], dtype=np.float64),
        }

    if "sesiones" in data or "default" in data:
        # Formato nuevo (multi-sesión).
        default = _a_arrays(data["default"]) if data.get("default") else None
        sesiones = {sid: _a_arrays(entrada) for sid, entrada in data.get("sesiones", {}).items()}
    else:
        # Formato plano (pre multi-sesión): única calibración = default.
        default = _a_arrays(data)
        sesiones = {}

    return {"default": default, "sesiones": sesiones}


def cargar_calibracion(ruta: str) -> dict:
    """Compatibilidad hacia atrás: retorna solo la calibración "default"
    (la más reciente guardada), ignorando cualquier indexación por sesión.
    Usar cargar_calibraciones() + normalizar_dataframe_multisesion() para
    datasets con más de una sesión de calibración."""
    calibraciones = cargar_calibraciones(ruta)
    if calibraciones["default"] is None:
        raise ValueError(f"El archivo de calibración no tiene una calibración 'default': {ruta}")
    return calibraciones["default"]


def normalizar_dataframe_multisesion(df: pd.DataFrame, calibraciones: dict) -> pd.DataFrame:
    """Normaliza `df` a %MVC aplicando, a cada fila, la calibración de su
    propia sesion_id (calibraciones["sesiones"][sesion_id]). Si `df` no
    tiene columna "sesion_id", o si alguna sesión no tiene calibración
    indexada, cae de vuelta a calibraciones["default"] para esas filas.

    Levanta ValueError si una sesión no tiene calibración indexada NI hay
    "default" disponible como respaldo — a diferencia de silenciarlo, para
    no normalizar datos con una calibración incorrecta sin que se note."""
    if "sesion_id" not in df.columns:
        if calibraciones["default"] is None:
            raise ValueError(
                "El dataset no tiene columna 'sesion_id' y la calibración "
                "no tiene 'default'. No hay con qué normalizar."
            )
        return normalizar_dataframe(df, calibraciones["default"])

    grupos_normalizados = []
    for sesion_id, grupo in df.groupby("sesion_id", sort=False, dropna=False):
        calib = calibraciones["sesiones"].get(sesion_id, calibraciones["default"])
        if calib is None:
            raise ValueError(
                f"La sesión '{sesion_id}' no tiene calibración indexada y "
                f"tampoco hay 'default' como respaldo. Recalibrar o guardar "
                f"una calibración 'default' antes de normalizar."
            )
        grupos_normalizados.append(normalizar_dataframe(grupo, calib))

    return pd.concat(grupos_normalizados).sort_index()


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
        calibraciones = cargar_calibraciones(args.calibracion)
        df_normalizado = normalizar_dataframe_multisesion(df, calibraciones)
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