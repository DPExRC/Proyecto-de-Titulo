# =============================================================================
# train.py — Entrenamiento: RF clasificador + RF regresor (pipeline dual)
# =============================================================================
# Uso:
#   python training/train.py
#   python training/train.py --csv data/datos_emg.csv
#
# Genera:
#   models/modelo_clasificador.pkl   RF clasificador  (REPOSO/FLEXION/EXTENSION)
#   models/modelo_regresor.pkl       RF regresor      (ángulo continuo 0–180°)
#   models/meta_entrenamiento.json   métricas de ambos modelos
#
# El clasificador usa las 4 features originales.
# El regresor usa las mismas 4 features — se entrena sobre todos los registros
# donde el ángulo no es 180° (reposo puro) para aprender la relación
# entre actividad muscular y posición angular.
# =============================================================================

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                              confusion_matrix, classification_report,
                              mean_absolute_error, r2_score)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from src.config import (DATA_PATH, NOMBRES_FEATURES, CLASES, N_CLASES,
                         MIN_REGISTROS_CLASE, ANGULO_MIN, ANGULO_MAX,
                         UMBRAL_REPOSO_PCT)

MODEL_DIR        = os.path.join(os.path.dirname(__file__), "..", "models")
PATH_CLASIF      = os.path.join(MODEL_DIR, "modelo_clasificador.pkl")
PATH_REGRESOR    = os.path.join(MODEL_DIR, "modelo_regresor.pkl")
PATH_META        = os.path.join(MODEL_DIR, "meta_entrenamiento.json")

# Umbral para derivar la clase desde el ángulo (usado solo en entrenamiento)
# 180°           → REPOSO    (clase 0)
# 91°–179°       → EXTENSION (clase 2)  — brazo en rango extendido activo
# 0°–90°         → FLEXION   (clase 1)  — brazo en rango de flexión
UMBRAL_FLEXION   = 90.0   # ángulos <= este valor → FLEXION
UMBRAL_EXTENSION = 170.0  # ángulos >= este valor → REPOSO


def angulo_a_clase(angulo: float) -> int:
    """Deriva la etiqueta de clase a partir del ángulo declarado."""
    if angulo >= UMBRAL_EXTENSION:
        return 0  # REPOSO
    elif angulo <= UMBRAL_FLEXION:
        return 1  # FLEXION
    else:
        return 2  # EXTENSION (rango intermedio hacia extensión)


def cargar_datos(csv_path: str):
    if not os.path.exists(csv_path):
        print(f"[train] Archivo no encontrado: {csv_path}")
        print("[train] Ejecuta data/captura.py primero.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"[train] Dataset: {len(df)} registros")

    # Validar columnas
    cols_req = NOMBRES_FEATURES + ["angulo"]
    faltantes = [c for c in cols_req if c not in df.columns]
    if faltantes:
        print(f"[train] Columnas faltantes: {faltantes}")
        sys.exit(1)

    # Eliminar filas con NaN
    df = df.dropna(subset=cols_req)
    print(f"[train] Registros válidos tras limpieza: {len(df)}")

    X = df[NOMBRES_FEATURES].values.astype(np.float32)
    y_angulo = df["angulo"].values.astype(np.float32)

    # Derivar clase desde ángulo
    y_clase = np.array([angulo_a_clase(a) for a in y_angulo], dtype=int)

    # Distribución
    print("\n[train] Distribución de ángulos:")
    angulos_unicos = sorted(df["angulo"].unique())
    for a in angulos_unicos:
        n = (df["angulo"] == a).sum()
        print(f"  {a:6.1f}°  →  {n:4d} registros  (clase {angulo_a_clase(a)}: {CLASES[angulo_a_clase(a)]})")

    print("\n[train] Distribución por clase derivada:")
    for c, nombre in CLASES.items():
        n = (y_clase == c).sum()
        alerta = "  ⚠ BAJO" if n < MIN_REGISTROS_CLASE else ""
        print(f"  Clase {c} ({nombre}): {n} registros{alerta}")

    return X, y_angulo, y_clase


def entrenar_clasificador(X, y_clase, seed, test_size):
    print("\n" + "="*55)
    print("  MODELO 1 — Clasificador (REPOSO / FLEXION / EXTENSION)")
    print("="*55)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_clase, test_size=test_size, stratify=y_clase, random_state=seed)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1
        ))
    ])

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_scores = cross_val_score(pipeline, X, y_clase, cv=cv, scoring="balanced_accuracy")
    print(f"[clasificador] CV balanced_accuracy: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)

    acc     = accuracy_score(y_te, y_pred)
    bal_acc = balanced_accuracy_score(y_te, y_pred)
    cm      = confusion_matrix(y_te, y_pred)

    print(f"[clasificador] Accuracy:          {acc:.4f}")
    print(f"[clasificador] Balanced accuracy: {bal_acc:.4f}")
    print(f"\n[clasificador] Matriz de confusión (filas=real, cols=predicho):")
    nombres = [CLASES[i] for i in range(N_CLASES)]
    print("         " + "  ".join(f"{n[:9]:>9}" for n in nombres))
    for i, fila in enumerate(cm):
        print(f"  {nombres[i][:9]:>9}  " + "  ".join(f"{v:>9}" for v in fila))
    print(f"\n{classification_report(y_te, y_pred, target_names=list(CLASES.values()), zero_division=0)}")

    imp = pipeline.named_steps["clf"].feature_importances_
    print("[clasificador] Importancia de features:")
    for nombre, v in sorted(zip(NOMBRES_FEATURES, imp), key=lambda x: x[1], reverse=True):
        print(f"  {nombre:<22} {v:.4f}  {'█'*int(v*40)}")

    meta = {
        "cv_balanced_acc_mean": float(cv_scores.mean()),
        "cv_balanced_acc_std":  float(cv_scores.std()),
        "test_accuracy":        float(acc),
        "test_balanced_acc":    float(bal_acc),
        "importancias":         dict(zip(NOMBRES_FEATURES, imp.tolist()))
    }
    return pipeline, meta


def entrenar_regresor(X, y_angulo, seed, test_size):
    print("\n" + "="*55)
    print("  MODELO 2 — Regresor (ángulo continuo 0°–180°)")
    print("="*55)

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_angulo, test_size=test_size, random_state=seed)

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
    cv_mae = -cross_val_score(pipeline, X, y_angulo, cv=cv, scoring="neg_mean_absolute_error")
    print(f"[regresor] CV MAE: {cv_mae.mean():.2f}° ± {cv_mae.std():.2f}°")

    pipeline.fit(X_tr, y_tr)
    y_pred = pipeline.predict(X_te)

    mae = mean_absolute_error(y_te, y_pred)
    r2  = r2_score(y_te, y_pred)

    print(f"[regresor] MAE test:  {mae:.2f}°")
    print(f"[regresor] R²  test:  {r2:.4f}")

    # Error por rango de ángulo
    print("[regresor] MAE por rango de ángulo:")
    rangos = [(0, 30), (30, 90), (90, 150), (150, 180)]
    for lo, hi in rangos:
        mask = (y_te >= lo) & (y_te <= hi)
        if mask.sum() > 0:
            mae_r = mean_absolute_error(y_te[mask], y_pred[mask])
            print(f"  [{lo:3d}°–{hi:3d}°]  n={mask.sum():4d}  MAE={mae_r:.2f}°")

    imp = pipeline.named_steps["reg"].feature_importances_
    print("[regresor] Importancia de features:")
    for nombre, v in sorted(zip(NOMBRES_FEATURES, imp), key=lambda x: x[1], reverse=True):
        print(f"  {nombre:<22} {v:.4f}  {'█'*int(v*40)}")

    meta = {
        "cv_mae_mean":    float(cv_mae.mean()),
        "cv_mae_std":     float(cv_mae.std()),
        "test_mae":       float(mae),
        "test_r2":        float(r2),
        "importancias":   dict(zip(NOMBRES_FEATURES, imp.tolist()))
    }
    return pipeline, meta


def main():
    parser = argparse.ArgumentParser(description="Entrenamiento dual: clasificador + regresor EMG")
    parser.add_argument("--csv",       default=DATA_PATH)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--seed",      type=int,   default=42)
    args = parser.parse_args()

    print("="*55)
    print("  Entrenamiento — Pipeline dual EMG")
    print(f"  Dataset: {args.csv}")
    print("="*55)

    X, y_angulo, y_clase = cargar_datos(args.csv)

    modelo_clasif, meta_clasif   = entrenar_clasificador(X, y_clase,  args.seed, args.test_size)
    modelo_regresor, meta_regres = entrenar_regresor(X, y_angulo, args.seed, args.test_size)

    # Guardar modelos
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(modelo_clasif,   PATH_CLASIF)
    joblib.dump(modelo_regresor, PATH_REGRESOR)
    print(f"\n[train] Clasificador guardado: {os.path.abspath(PATH_CLASIF)}")
    print(f"[train] Regresor guardado:     {os.path.abspath(PATH_REGRESOR)}")

    # Guardar metadatos
    meta = {
        "n_registros":   int(len(X)),
        "test_size":     args.test_size,
        "seed":          args.seed,
        "features":      NOMBRES_FEATURES,
        "clasificador":  meta_clasif,
        "regresor":      meta_regres
    }
    with open(PATH_META, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"[train] Metadatos guardados:   {os.path.abspath(PATH_META)}")

    # Advertencias finales
    if meta_clasif["test_balanced_acc"] < 0.80:
        print("\n[train] ⚠  Clasificador: balanced_accuracy < 0.80 — capturar más datos.")
    if meta_regres["test_mae"] > 15.0:
        print(f"[train] ⚠  Regresor: MAE = {meta_regres['test_mae']:.1f}° > 15° — capturar más ángulos o rondas.")


if __name__ == "__main__":
    main()