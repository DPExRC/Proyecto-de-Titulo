# =============================================================================
# train.py — Entrenamiento: RF Regresor multi-salida (codo + muñeca)
# =============================================================================
# Uso independiente:
#   python training/train.py
#   python training/train.py --csv data/datos_emg_normalizado.csv
#
# Uso desde main.py (flujo "Entrenar"):
#   from training.train import entrenar_pipeline
#   pipeline, meta = entrenar_pipeline(csv_path)
#
# CAMBIO respecto a la versión anterior: este script ahora entrena sobre
# las columnas NORMALIZADAS (%MVC, prefijo "pct_mvc_") generadas por
# estandarizacion.py, no sobre las columnas crudas de NOMBRES_FEATURES.
# Antes leía df[NOMBRES_FEATURES] directamente, que son los valores
# crudos en mV — la calibración baseline/MVC nunca llegaba a influir en
# el modelo. Si el CSV no tiene columnas "pct_mvc_*" (por ejemplo, si
# corres esto sobre datos_emg.csv crudo sin pasar por estandarizacion.py
# primero), cae de vuelta a las columnas crudas con una advertencia
# explícita — útil para pruebas rápidas, pero NO recomendado para el
# modelo final.
#
# Arquitectura del sistema (v3.0, confirmada):
#   - 3 canales sEMG: bíceps braquial, tríceps braquial, pronator teres (antebrazo)
#   - 2 DOF controlados por EMG, ambos con reposo = 0°:
#       DOF 1 — Codo: bidireccional, par antagonista bíceps/tríceps.
#               Bíceps incrementa el ángulo (flexión); tríceps acelera el
#               retorno hacia 0° (el rango no baja de 0°).
#       DOF 2 — Muñeca: unidireccional, canal único pronator teres (antebrazo).
#   - Vector de 12 features (RMS, MAV, WL, ZCR por canal), definido en
#     config.py — no se hardcodea aquí.
#   - Un único RandomForestRegressor multi-salida predice angulo_codo y
#     angulo_muneca simultáneamente. Sin etapa de clasificación: el
#     gating de reposo/ruido se resuelve en el firmware/predictor
#     mediante UMBRAL_BAJO/UMBRAL_ALTO, el filtro exponencial asimétrico
#     y el limitador de slew-rate — no es responsabilidad de este script.
#
# Genera:
#   models/modelo_regresor.pkl       RF regresor multi-salida (Pipeline
#                                     completo: StandardScaler + RF)
#   models/meta_entrenamiento.json   métricas del modelo, por DOF
#
# Requiere que el CSV de entrada tenga las columnas "pct_mvc_<feature>"
# (salida de estandarizacion.py) o, en su defecto, NOMBRES_FEATURES
# crudos (con advertencia), más COL_ANGULO_CODO y COL_ANGULO_MUNECA.
# =============================================================================

# =============================================================================
# train.py — Entrenamiento: RF Regresor multi-salida (Visual Rich)
# =============================================================================

import os
import sys
import json
import argparse
import time
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, KFold, cross_validate
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.metrics import mean_squared_error

from emg_arm.config import (NOMBRES_FEATURES, COLS_TARGET, COL_ANGULO_CODO,
                         COL_ANGULO_MUNECA, DATA_PATH)

# Importaciones de la estética unificada Rich
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich import box

console = Console()

DATA_PATH_NORMALIZADO = os.path.join(
    os.path.dirname(__file__), "..", "data", "datos_emg_normalizado.csv"
)
PATH_MODELO = os.path.join(
    os.path.dirname(__file__), "..", "models", "modelo_regresor.pkl"
)
PATH_META = os.path.join(
    os.path.dirname(__file__), "..", "models", "meta_entrenamiento.json"
)
# Historial acumulativo: cada entrenamiento se AGREGA (no sobreescribe), para
# poder auditar la evolución del modelo entre corridas — meta_entrenamiento.json
# solo guarda el último resultado; este archivo guarda todos.
PATH_HISTORIAL = os.path.join(
    os.path.dirname(__file__), "..", "models", "historial_entrenamientos.jsonl"
)


def cargar_datos(csv_path: str):
    df = pd.read_csv(csv_path)
    
    # Buscamos si existen las columnas normalizadas en %MVC
    cols_pct = [f"pct_mvc_{f}" for f in NOMBRES_FEATURES]
    usando_normalizadas = all(c in df.columns for c in cols_pct)

    if usando_normalizadas:
        cols_X = cols_pct
        tipo_features = "[bold green]%MVC (Estandarizado offline)[/]"
    else:
        # Caída de respaldo automática si no se pasó por estandarizacion.py
        cols_X = NOMBRES_FEATURES
        tipo_features = "[bold yellow]CRUDAS en mV (¡No recomendado para modelo final!)[/]"
        console.print(
            "\n[bold yellow]⚠ [train] ADVERTENCIA:[/] El CSV no contiene columnas 'pct_mvc_*'. "
            "Se entrenará usando las features crudas.", style="yellow"
        )

    for col in COLS_TARGET:
        if col not in df.columns:
            raise KeyError(f"La columna target '{col}' no existe en el dataset.")

    X = df[cols_X].to_numpy(dtype=np.float64)
    y = df[COLS_TARGET].to_numpy(dtype=np.float64)

    # Mostrar resumen del estado de los datos en un panel compacto
    info_datos = (
        f"Registros totales en disco:   [bold green]{len(df)}[/]\n"
        f"Representación de Entrada:     {tipo_features}\n"
        f"Dimensión de la Matriz X:      [dim]{X.shape[0]} filas x {X.shape[1]} columnas[/]\n"
        f"Variables de Salida (Targets): [dim]{COLS_TARGET}[/]"
    )
    console.print(Panel(info_datos, title="[bold cyan]Análisis de Datos de Entrada[/]", border_style="cyan", expand=False))

    return X, y, cols_X, df


def verificar_distribucion_targets(df: pd.DataFrame):
    """Genera un reporte analítico de rangos y salta una alerta visual si

    el set carece de datos en la fase de reposo (codo cerca de 0°)."""
    # Tabla de consistencia geométrica
    tabla = Table(title="\nResumen de Distribución Geométrica (Targets)", box=box.ROUNDED, border_style="dim")
    tabla.add_column("Grado de Libertad (DOF)", style="cyan")
    tabla.add_column("Mínimo", justify="right")
    tabla.add_column("Máximo", justify="right")
    tabla.add_column("Media", justify="right")
    tabla.add_column("Desviación Std", justify="right")

    for col in COLS_TARGET:
        valores = df[col].to_numpy()
        tabla.add_row(
            col,
            f"{np.min(valores):.1f}°",
            f"{np.max(valores):.1f}°",
            f"{np.mean(valores):.1f}°",
            f"{np.std(valores):.1f}°"
        )
    console.print(tabla)

    # Alerta crítica de balance de datos
    codo_valores = df[COL_ANGULO_CODO].to_numpy()
    if not np.any(codo_valores < 15.0):
        msg_alerta = (
            "[bold red]CRÍTICO:[/] Ningún registro de '[bold]angulo_codo[/]' está cerca de 0° (reposo).\n\n"
            "[dim]El RandomForest no sabrá qué hacer cuando relajes el brazo en producción.\n"
            "Por favor, vuelve al menú, selecciona 'Capturar' y añade muestras manteniendo\n"
            "ambos motores en la posición '0,0'.[/]"
        )
        console.print("\n", Panel(msg_alerta, title="[bold yellow]¡Falta Fase de Reposo en el Dataset![/]", border_style="yellow", expand=False))


def evaluar_por_rangos(y_test, y_pred, col_name, rangos: list):
    """Calcula el MAE local dividiendo el espacio de predicción en sub-rangos."""
    idx_col = COLS_TARGET.index(col_name)
    y_t = y_test[:, idx_col]
    y_p = y_pred[:, idx_col]

    lineas_reporte = []
    for (r_min, r_max) in rangos:
        mascara = (y_t >= r_min) & (y_t <= r_max)
        n_sub = int(np.sum(mascara))
        if n_sub > 0:
            mae_sub = np.mean(np.abs(y_t[mascara] - y_p[mascara]))
            lineas_reporte.append(f"      [{r_min:3d}°–{r_max:3d}°]  n={n_sub:4d}  [bold yellow]MAE={mae_sub:5.2f}°[/]")
    return "\n".join(lineas_reporte)


def entrenar_pipeline(csv_path: str, test_size: float = 0.2, seed: int = 42):
    X, y, cols_X, df = cargar_datos(csv_path)
    verificar_distribucion_targets(df)

    # 1. Separación Hold-Out (Evaluación final limpia)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    # Definición del modelo base de regresión paralela
    regresor_base = RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=seed,
        n_jobs=1
    )

    # Un RF independiente por DOF: codo (features de bíceps/tríceps) y
    # muñeca (features de antebrazo) son casi disjuntas por diseño, así que
    # un único árbol optimizando el MSE combinado de ambas salidas diluye
    # la calidad de las particiones para las dos. n_jobs aquí paraleliza
    # entre los dos DOF; el n_jobs=1 del RF base evita el problema de
    # sobrecarga de joblib ya identificado en inferencia de una fila a 50 Hz.
    regresor = MultiOutputRegressor(regresor_base, n_jobs=1)

    # Encapsulamos el pipeline listo para producción
    pipeline = Pipeline([
        ("reg", regresor)
    ])

    console.print("\n[cyan]⚙ Ejecutando Validación Cruzada (5-Fold CV)...[/]")
    cv = KFold(n_splits=5, shuffle=True, random_state=seed)

    # Resultados CRUDOS por fold — esto es lo que llena la Tabla 6.7 completa
    # (antes solo se guardaba el promedio; cada fold se pierde al cerrar la
    # terminal). Se guarda MAE por DOF + R2 promedio de ambos DOF, por fold.
    fold_resultados = []   # lista de dicts, uno por fold
    mae_cv_por_col = {col: [] for col in COLS_TARGET}
    for i_fold, (train_idx, val_idx) in enumerate(cv.split(X_train), start=1):
        # clone() reproduce exactamente la arquitectura y los hiperparámetros
        # de 'regresor' (el mismo MultiOutputRegressor de 200 árboles/DOF que
        # se despliega más abajo). Antes se usaba aquí un RF de 30 árboles
        # con n_jobs=-1, distinto del modelo final — la Tabla 6.7 estaba
        # validando una configuración que no era la que terminaba en
        # producción.
        p_temporal = Pipeline([("reg", clone(regresor))])
        p_temporal.fit(X_train[train_idx], y_train[train_idx])
        preds_val = p_temporal.predict(X_train[val_idx])

        mae_fold = {}
        r2_fold = {}
        for idx_col, col in enumerate(COLS_TARGET):
            y_v = y_train[val_idx, idx_col]
            y_p = preds_val[:, idx_col]
            mae_f = float(np.mean(np.abs(y_v - y_p)))
            mae_cv_por_col[col].append(mae_f)
            mae_fold[col] = mae_f

            ss_res = np.sum((y_v - y_p) ** 2)
            ss_tot = np.sum((y_v - np.mean(y_v)) ** 2)
            r2_fold[col] = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0

        r2_promedio_fold = float(np.mean(list(r2_fold.values())))
        fold_resultados.append({
            "fold": i_fold,
            "mae_codo": mae_fold.get(COL_ANGULO_CODO),
            "mae_muneca": mae_fold.get(COL_ANGULO_MUNECA),
            "r2_codo": r2_fold.get(COL_ANGULO_CODO),
            "r2_muneca": r2_fold.get(COL_ANGULO_MUNECA),
            "r2_promedio": r2_promedio_fold,
        })

    # --- TABLA 6.7 en consola: detalle por fold + media ± desv. est. -----------
    tabla_cv = Table(title="\nValidación Cruzada 5-Fold — Detalle por Partición (Tabla 6.7)",
                      box=box.ROUNDED, border_style="dim")
    tabla_cv.add_column("Fold", justify="center", style="bold cyan")
    tabla_cv.add_column("MAE Codo (°)", justify="right")
    tabla_cv.add_column("MAE Muñeca (°)", justify="right")
    tabla_cv.add_column("R² promedio", justify="right")
    for fr in fold_resultados:
        tabla_cv.add_row(str(fr["fold"]), f"{fr['mae_codo']:.2f}", f"{fr['mae_muneca']:.2f}", f"{fr['r2_promedio']:.4f}")

    _mae_codo_arr = np.array([fr["mae_codo"] for fr in fold_resultados])
    _mae_muneca_arr = np.array([fr["mae_muneca"] for fr in fold_resultados])
    _r2_prom_arr = np.array([fr["r2_promedio"] for fr in fold_resultados])
    tabla_cv.add_row(
        "Media±DE",
        f"{_mae_codo_arr.mean():.2f}±{_mae_codo_arr.std():.2f}",
        f"{_mae_muneca_arr.mean():.2f}±{_mae_muneca_arr.std():.2f}",
        f"{_r2_prom_arr.mean():.4f}±{_r2_prom_arr.std():.4f}",
        style="bold"
    )
    console.print(tabla_cv)

    # 2. Ajuste Final sobre el conjunto de entrenamiento completo
    console.print("[cyan]⚙ Ajustando modelo definitivo sobre el set de entrenamiento...[/]")
    t_start = time.time()
    pipeline.fit(X_train, y_train)
    t_compilación = time.time() - t_start

    # 3. Inferencia sobre el conjunto Hold-out de evaluación
    y_pred = pipeline.predict(X_test)

    # --- TABLA 6.8: MAE, RMSE y R² sobre el conjunto de prueba -----------------
    tabla_perf = Table(title=f"\nMétricas del Regresor Multi-Salida (Hold-Out {test_size*100:.0f}%) — Tabla 6.8", box=box.ROUNDED)
    tabla_perf.add_column("Grado de Libertad (DOF)", style="bold cyan")
    tabla_perf.add_column("CV MAE (Train)", justify="right", style="green")
    tabla_perf.add_column("Test MAE", justify="right", style="bold magenta")
    tabla_perf.add_column("Test RMSE", justify="right", style="bold magenta")
    tabla_perf.add_column("Coeficiente R²", justify="right")

    meta_modelo = {"cv_5fold": {}, "test": {}}

    for i, col in enumerate(COLS_TARGET):
        mae_cv = np.mean(mae_cv_por_col[col])
        mae_test = np.mean(np.abs(y_test[:, i] - y_pred[:, i]))
        rmse_test = float(np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i])))

        # Coeficiente de determinación R² manual por columna
        ss_res = np.sum((y_test[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_test[:, i] - np.mean(y_test[:, i])) ** 2)
        r2_col = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 1.0

        tabla_perf.add_row(col, f"{mae_cv:.2f}°", f"{mae_test:.2f}°", f"{rmse_test:.2f}°", f"{r2_col:.4f}")

        # Guardar en diccionario de metadatos
        meta_modelo["cv_5fold"][col] = {"mae_promedio": float(mae_cv)}
        meta_modelo["test"][col] = {
            "test_mae": float(mae_test),
            "test_rmse": rmse_test,
            "test_r2": float(r2_col)
        }
    
    console.print(tabla_perf)

    # --- TABLA 6.9: Comparación Entrenamiento vs. Prueba (sobreajuste) --------
    y_train_pred = pipeline.predict(X_train)
    tabla_overfit = Table(title="\nAnálisis de Sobreajuste — Entrenamiento vs. Prueba (Tabla 6.9)",
                           box=box.ROUNDED, border_style="dim")
    tabla_overfit.add_column("Conjunto", style="bold cyan")
    tabla_overfit.add_column("MAE Codo (°)", justify="right")
    tabla_overfit.add_column("MAE Muñeca (°)", justify="right")
    tabla_overfit.add_column("R²", justify="right")

    meta_modelo["train"] = {}
    meta_modelo["brecha"] = {}
    mae_train_por_col, r2_train_por_col = {}, {}
    for i, col in enumerate(COLS_TARGET):
        mae_tr = float(np.mean(np.abs(y_train[:, i] - y_train_pred[:, i])))
        ss_res = np.sum((y_train[:, i] - y_train_pred[:, i]) ** 2)
        ss_tot = np.sum((y_train[:, i] - np.mean(y_train[:, i])) ** 2)
        r2_tr = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 1.0
        mae_train_por_col[col] = mae_tr
        r2_train_por_col[col] = r2_tr
        meta_modelo["train"][col] = {"train_mae": mae_tr, "train_r2": r2_tr}
        meta_modelo["brecha"][col] = {
            "mae": meta_modelo["test"][col]["test_mae"] - mae_tr,
            "r2": meta_modelo["test"][col]["test_r2"] - r2_tr,
        }

    tabla_overfit.add_row("Entrenamiento",
                           f"{mae_train_por_col[COL_ANGULO_CODO]:.2f}",
                           f"{mae_train_por_col[COL_ANGULO_MUNECA]:.2f}",
                           f"{np.mean(list(r2_train_por_col.values())):.4f}")
    tabla_overfit.add_row("Prueba",
                           f"{meta_modelo['test'][COL_ANGULO_CODO]['test_mae']:.2f}",
                           f"{meta_modelo['test'][COL_ANGULO_MUNECA]['test_mae']:.2f}",
                           f"{np.mean([meta_modelo['test'][c]['test_r2'] for c in COLS_TARGET]):.4f}")
    tabla_overfit.add_row("Diferencia (brecha)",
                           f"{meta_modelo['brecha'][COL_ANGULO_CODO]['mae']:+.2f}",
                           f"{meta_modelo['brecha'][COL_ANGULO_MUNECA]['mae']:+.2f}",
                           f"{np.mean([meta_modelo['brecha'][c]['r2'] for c in COLS_TARGET]):+.4f}",
                           style="bold yellow")
    console.print(tabla_overfit)

    # --- Tiempo de inferencia PC real, medido muestra a muestra ----------------
    # No es una cifra sintética: se mide el predict() del pipeline YA
    # entrenado, fila por fila del propio set de prueba, con
    # time.perf_counter(). Esto es lo que llena la columna "Inferencia PC"
    # de la Tabla 8.2 con datos reales del hardware donde se entrena.
    tiempos_inferencia_ms = []
    for fila in X_test:
        t0 = time.perf_counter()
        pipeline.predict(fila.reshape(1, -1))
        tiempos_inferencia_ms.append((time.perf_counter() - t0) * 1000.0)
    tiempos_inferencia_ms = np.array(tiempos_inferencia_ms)

    meta_modelo["inferencia_pc"] = {
        "promedio_ms": float(tiempos_inferencia_ms.mean()),
        "desv_est_ms": float(tiempos_inferencia_ms.std()),
        "maximo_ms": float(tiempos_inferencia_ms.max()),
        "n_muestras": int(len(tiempos_inferencia_ms)),
    }
    console.print(
        f"\n[bold cyan]⏱ Inferencia PC (por muestra, n={len(tiempos_inferencia_ms)}):[/] "
        f"{tiempos_inferencia_ms.mean():.2f} ms (±{tiempos_inferencia_ms.std():.2f} ms), "
        f"máx {tiempos_inferencia_ms.max():.2f} ms"
    )

    meta_modelo["cv_5fold_por_fold"] = fold_resultados

    # --- REPORTE DE RESOLUCIÓN POR RANGOS CINEMÁTICOS ---
    console.print("\n[bold cyan]🔍 Análisis de Precisión Local por Segmentos:[/]")
    reporte_codo = evaluar_por_rangos(y_test, y_pred, COL_ANGULO_CODO, [(0, 30), (30, 90), (90, 150), (150, 180)])
    reporte_muneca = evaluar_por_rangos(y_test, y_pred, COL_ANGULO_MUNECA, [(0, 30), (30, 90), (90, 150)])
    
    if reporte_codo:
        console.print(f"  • [bold]{COL_ANGULO_CODO}:[/]\n{reporte_codo}")
    if reporte_muneca:
        console.print(f"  • [bold]{COL_ANGULO_MUNECA}:[/]\n{reporte_muneca}")

    # --- TABLA DE RELEVANCIA DE CARACTERÍSTICAS (IMPORTANCIAS DE BOSQUE) ---
    # rf_interno es un MultiOutputRegressor: un RF independiente por DOF en
    # rf_interno.estimators_ (mismo orden que COLS_TARGET), sin
    # feature_importances_ propio a nivel del wrapper.
    rf_interno = pipeline.named_steps["reg"]
    for nombre_dof, estimador_dof in zip(COLS_TARGET, rf_interno.estimators_):
        importancias = estimador_dof.feature_importances_

        tabla_imp = Table(title=f"\nImportancia de Características — {nombre_dof}", box=None)
        tabla_imp.add_column("Feature Normalizada (%MVC)", style="dim")
        tabla_imp.add_column("Peso", justify="right", style="bold green")
        tabla_imp.add_column("Distribución de Relevancia Visual")

        indices_ordenados = np.argsort(importancias)[::-1]
        for idx in indices_ordenados:
            peso = importancias[idx]
            # Generar barra gráfica con caracteres de bloques proporcionales
            barra = "█" * int(peso * 50)
            tabla_imp.add_row(cols_X[idx], f"{peso:.4f}", f"[magenta]{barra}[/]")

        console.print(tabla_imp)

    # Guardar artefactos binarios en disco
    os.makedirs(os.path.dirname(os.path.abspath(PATH_MODELO)), exist_ok=True)
    joblib.dump(pipeline, PATH_MODELO)
    console.print(f"\n[bold green]✓[/] Regresor (.pkl) exportado con Joblib en: [dim]{os.path.abspath(PATH_MODELO)}[/]")

    # Estructura final del JSON de metadatos
    meta = {
        "fecha_entrenamiento": time.strftime("%Y-%m-%d %H:%M:%S"),
        "tiempo_ajuste_segundos": float(t_compilación),
        "dataset_origen": os.path.abspath(csv_path),
        "hiperparametros": {
            "n_estimadores": regresor_base.n_estimators,
            "max_depth": regresor_base.max_depth,
            "min_samples_leaf": regresor_base.min_samples_leaf
        },
        "features_entrada": cols_X,
        "targets_salida": COLS_TARGET,
        "regresor": meta_modelo,
    }
    
    with open(PATH_META, "w") as f:
        json.dump(meta, f, indent=2)
    console.print(f"[bold green]✓[/] Metadatos de auditoría guardados en: [dim]{os.path.abspath(PATH_META)}[/]")

    # Historial ACUMULATIVO (append, nunca sobreescribe) — a diferencia de
    # meta_entrenamiento.json que solo guarda la última corrida, esto permite
    # auditar cómo evolucionaron las métricas entre entrenamientos sucesivos.
    os.makedirs(os.path.dirname(os.path.abspath(PATH_HISTORIAL)), exist_ok=True)
    with open(PATH_HISTORIAL, "a") as f:
        f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    console.print(f"[bold green]✓[/] Corrida añadida al historial: [dim]{os.path.abspath(PATH_HISTORIAL)}[/]\n")

    # Alertas finales de tolerancia si algún DOF supera desviaciones aceptables
    for col in COLS_TARGET:
        mae_test = meta_modelo["test"][col]["test_mae"]
        if mae_test > 15.0:
            console.print(
                Panel(f"[bold red]⚠ ALERTA DE PRECISIÓN EN {col.upper()}:[/] El error medio (MAE = {mae_test:.1f}°) "
                      f"supera los 15° de tolerancia.\n[dim]Se recomienda capturar más variedad de posiciones "
                      f"o revisar el ruido electromagnético de los electrodos.[/]", 
                      border_style="red", expand=False)
            )

    return pipeline, meta


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento del regresor EMG multi-salida")
    parser.add_argument("--csv", default=DATA_PATH_NORMALIZADO,
                         help="CSV normalizado (salida de estandarizacion.py)")
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Cabecera estilizada idéntica a main.py
    console.print("\n" + "=" * 60)
    console.print("  [bold magenta]TRAINING PIPELINE — Regresor Multi-Salida (RandomForest)[/]")
    console.print("=" * 60)

    csv_path = args.csv
    if not os.path.exists(csv_path) and os.path.exists(DATA_PATH):
        console.print(
            f"[bold yellow]⚠ [train] ALERTA:[/] No se encontró el dataset normalizado en: [dim]{csv_path}[/]\n"
            f"Se cae de vuelta al archivo crudo: [dim]{DATA_PATH}[/]"
        )
        csv_path = DATA_PATH
    elif not os.path.exists(csv_path) and not os.path.exists(DATA_PATH):
        console.print(f"[bold red]✗ Error crítico:[/] No existe ningún archivo de datos en {csv_path} ni en {DATA_PATH}")
        sys.exit(1)

    entrenar_pipeline(csv_path, test_size=args.test_size, seed=args.seed)


if __name__ == "__main__":
    main()