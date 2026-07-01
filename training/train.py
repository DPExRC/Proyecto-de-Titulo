# =============================================================================
# train.py — Entrenamiento: RF Regresor multi-salida (codo + muñeca)
# =============================================================================
# Uso:
#   python training/train.py
#   python training/train.py --csv data/datos_emg.csv
#
# Arquitectura del sistema (v3.0, confirmada):
#   - 3 canales sEMG: bíceps braquial, tríceps braquial, pronator teres (antebrazo)
#   - 2 DOF controlados por EMG, ambos con reposo = 0°:
#       DOF 1 — Codo: bidireccional, par antagonista bíceps/tríceps.
#               Bíceps incrementa el ángulo (flexión); tríceps acelera el
#               retorno hacia 0° (el rango no baja de 0°).
#       DOF 2 — Muñeca: unidireccional, canal único pronator teres (antebrazo).
#   - Vector de 12 features (RMS, MAV, WL, ZC por canal), definido en
#     config.py — no se hardcodea aquí.
#   - Un único RandomForestRegressor multi-salida predice angulo_codo y
#     angulo_muneca simultáneamente. Sin etapa de clasificación: el
#     gating de reposo/ruido se resuelve en el firmware/predictor
#     mediante UMBRAL_BAJO/UMBRAL_ALTO, el filtro exponencial asimétrico
#     y el limitador de slew-rate — no es responsabilidad de este script.
#
# Genera:
#   models/modelo_regresor.pkl       RF regresor multi-salida
#   models/meta_entrenamiento.json   métricas del modelo, por DOF
#
# Requiere que el CSV de entrada (generado por data/captura.py) tenga
# las 12 columnas de NOMBRES_FEATURES definidas en config.py, más las
# columnas COL_ANGULO_CODO y COL_ANGULO_MUNECA. Si captura.py todavía no
# registra ambos ángulos por separado (heredado de una versión anterior
# de 1 solo DOF), debe actualizarse antes de poder usar este script.
# =============================================================================

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, KFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (DATA_PATH, NOMBRES_FEATURES, N_FEATURES,
                         COLS_TARGET, ANGULO_MIN, ANGULO_MAX)  # noqa: E402

MODEL_DIR     = os.path.join(os.path.dirname(__file__), "..", "models")
PATH_REGRESOR = os.path.join(MODEL_DIR, "modelo_regresor.pkl")
PATH_META     = os.path.join(MODEL_DIR, "meta_entrenamiento.json")


def cargar_datos(csv_path: str):
    if not os.path.exists(csv_path):
        print(f"[train] Archivo no encontrado: {csv_path}")
        print("[train] Ejecuta data/captura.py primero.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"[train] Dataset: {len(df)} registros")

    if len(NOMBRES_FEATURES) != N_FEATURES:
        print(f"[train] ADVERTENCIA: config.py declara N_FEATURES="
              f"{N_FEATURES} pero NOMBRES_FEATURES tiene "
              f"{len(NOMBRES_FEATURES)} entradas. Revisar config.py.")

    cols_req = NOMBRES_FEATURES + COLS_TARGET
    faltantes = [c for c in cols_req if c not in df.columns]
    if faltantes:
        print(f"[train] Columnas faltantes en el CSV: {faltantes}")
        print("[train] Verificar que data/captura.py genere estas "
              "columnas exactamente (ver config.py: NOMBRES_FEATURES, "
              "COL_ANGULO_CODO, COL_ANGULO_MUNECA).")
        sys.exit(1)

    df = df.dropna(subset=cols_req)
    print(f"[train] Registros válidos tras limpieza: {len(df)}")

    for col in COLS_TARGET:
        fuera_rango = ~df[col].between(ANGULO_MIN, ANGULO_MAX)
        if fuera_rango.any():
            print(f"[train] ADVERTENCIA: {fuera_rango.sum()} registros con "
                  f"'{col}' fuera de [{ANGULO_MIN}, {ANGULO_MAX}]. Se descartan.")
            df = df[~fuera_rango]

    X = df[NOMBRES_FEATURES].values.astype(np.float32)
    y = df[COLS_TARGET].values.astype(np.float32)  # shape (n, 2)

    print("\n[train] Resumen de ángulos objetivo:")
    for i, col in enumerate(COLS_TARGET):
        print(f"  {col:<15}  min={y[:, i].min():6.1f}°  "
              f"max={y[:, i].max():6.1f}°  "
              f"media={y[:, i].mean():6.1f}°  "
              f"std={y[:, i].std():6.1f}°")

    # Verificación de cordura sobre la convención reposo=0°: si ninguna
    # muestra cae cerca de 0° en algún DOF, probablemente falta capturar
    # la fase de reposo para ese canal.
    for i, col in enumerate(COLS_TARGET):
        cerca_de_cero = (y[:, i] <= 10.0).sum()
        if cerca_de_cero == 0:
            print(f"[train] ADVERTENCIA: ningún registro de '{col}' está "
                  f"cerca de 0° (reposo). Verificar que la captura incluya "
                  f"la fase de reposo para este DOF.")

    return X, y


def entrenar_regresor(X, y, seed, test_size):
    print("\n" + "=" * 60)
    print("  Regresor multi-salida — Codo (bidireccional) + "
          "Muñeca (unidireccional)")
    print("=" * 60)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=seed)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", RandomForestRegressor(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            random_state=seed,
            n_jobs=-1
        ))
    ])

    cv = KFold(n_splits=5, shuffle=True, random_state=seed)
    y_cv_pred = cross_val_predict(pipeline, X, y, cv=cv)

    print("[regresor] Validación cruzada (5-fold):")
    cv_metricas = {}
    for i, col in enumerate(COLS_TARGET):
        mae_cv = mean_absolute_error(y[:, i], y_cv_pred[:, i])
        r2_cv = r2_score(y[:, i], y_cv_pred[:, i])
        print(f"  {col:<15}  CV MAE = {mae_cv:6.2f}°   CV R² = {r2_cv:.4f}")
        cv_metricas[col] = {"cv_mae": float(mae_cv), "cv_r2": float(r2_cv)}

    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)

    print("\n[regresor] Conjunto de evaluación (hold-out):")
    test_metricas = {}
    for i, col in enumerate(COLS_TARGET):
        mae = mean_absolute_error(y_te[:, i], y_pred[:, i])
        r2 = r2_score(y_te[:, i], y_pred[:, i])
        print(f"  {col:<15}  MAE = {mae:6.2f}°   R² = {r2:.4f}")
        test_metricas[col] = {"test_mae": float(mae), "test_r2": float(r2)}

        print(f"    MAE por rango de ángulo ({col}):")
        rangos = [(0, 30), (30, 90), (90, 150), (150, 180)]
        for lo, hi in rangos:
            mask = (y_te[:, i] >= lo) & (y_te[:, i] <= hi)
            if mask.sum() > 0:
                mae_r = mean_absolute_error(y_te[mask, i], y_pred[mask, i])
                print(f"      [{lo:3d}°–{hi:3d}°]  n={mask.sum():4d}  "
                      f"MAE={mae_r:.2f}°")

    imp = pipeline.named_steps["reg"].feature_importances_
    print("\n[regresor] Importancia de features (agregada sobre ambos DOF):")
    for nombre, v in sorted(zip(NOMBRES_FEATURES, imp),
                             key=lambda x: x[1], reverse=True):
        print(f"  {nombre:<22} {v:.4f}  {'█' * int(v * 40)}")

    meta = {
        "cv": cv_metricas,
        "test": test_metricas,
        "importancias": dict(zip(NOMBRES_FEATURES, imp.tolist())),
    }
    return pipeline, meta


def main():
    parser = argparse.ArgumentParser(
        description="Entrenamiento del regresor EMG multi-salida "
                     "(codo + muñeca)")
    parser.add_argument("--csv", default=DATA_PATH)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("=" * 60)
    print("  Entrenamiento — Regresor multi-salida EMG (2 DOF, 3 canales)")
    print(f"  Dataset: {args.csv}")
    print("=" * 60)

    X, y = cargar_datos(args.csv)
    modelo, meta_modelo = entrenar_regresor(X, y, args.seed, args.test_size)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(modelo, PATH_REGRESOR)
    print(f"\n[train] Regresor guardado: {os.path.abspath(PATH_REGRESOR)}")

    meta = {
        "n_registros": int(len(X)),
        "test_size": args.test_size,
        "seed": args.seed,
        "features": NOMBRES_FEATURES,
        "targets": COLS_TARGET,
        "regresor": meta_modelo,
    }
    with open(PATH_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] Metadatos guardados: {os.path.abspath(PATH_META)}")

    for col in COLS_TARGET:
        mae_test = meta_modelo["test"][col]["test_mae"]
        if mae_test > 15.0:
            print(f"[train] ⚠  {col}: MAE = {mae_test:.1f}° > 15° — "
                  f"capturar más datos o revisar calidad de señal.")


if __name__ == "__main__":
    main()