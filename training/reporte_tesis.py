# =============================================================================
# reporte_tesis.py — Vuelca a texto las Tablas 6.6, 6.7, 6.8 y 6.9 del informe
# =============================================================================
# Lee ÚNICAMENTE archivos ya persistidos en disco (no recalcula nada):
#   - data/sesiones.json              -> Tabla 6.6 (composición del dataset)
#   - models/meta_entrenamiento.json  -> Tablas 6.7, 6.8, 6.9 + Inferencia PC
#
# Uso:
#   python training/reporte_tesis.py
#
# Si algún archivo no existe todavía, el script lo indica en vez de fallar,
# para que sepas exactamente qué paso del flujo (Capturar / Entrenar) falta
# correr antes de poder completar esa tabla.
# =============================================================================

import os
import sys
import json

RAIZ = os.path.join(os.path.dirname(__file__), "..")
RUTA_SESIONES = os.path.join(RAIZ, "data", "sesiones.json")
RUTA_META = os.path.join(RAIZ, "models", "meta_entrenamiento.json")


def _cargar_json(ruta):
    if not os.path.exists(ruta):
        return None
    with open(ruta, "r") as f:
        return json.load(f)


def tabla_6_6():
    sesiones = _cargar_json(RUTA_SESIONES)
    print("\n=== Tabla 6.6: Composición del conjunto de datos ===")
    if not sesiones:
        print("  [FALTA] No existe data/sesiones.json todavía.")
        print("  -> Corre 'python data/capture.py' (o el modo Entrenar de main.py)"
              " al menos una vez.")
        return

    n_sesiones = len(sesiones)
    total_muestras = sum(s["total_vectores"] for s in sesiones)
    total_repeticiones = sum(s["total_repeticiones"] for s in sesiones)
    duracion_total_s = sum(s["duracion_total_s"] for s in sesiones)

    posiciones = set()
    for s in sesiones:
        posiciones.update(s["repeticiones_por_posicion"].keys())

    print(f"  Número de sesiones de captura:                    {n_sesiones}")
    print(f"  Posiciones angulares de referencia distintas:     {len(posiciones)}  {sorted(posiciones)}")
    print(f"  Total de repeticiones (posición x sesión):        {total_repeticiones}")
    print(f"  Total de muestras (vectores de 12D etiquetados):  {total_muestras}")
    print(f"  Duración total de captura (todas las sesiones):   {duracion_total_s:.1f} s "
          f"({duracion_total_s/60:.1f} min)")
    print(f"  Duración promedio por sesión:                     {duracion_total_s/n_sesiones:.1f} s")


def tabla_6_7(meta):
    print("\n=== Tabla 6.7: Validación cruzada 5-fold (conjunto de entrenamiento) ===")
    folds = meta.get("regresor", {}).get("cv_5fold_por_fold")
    if not folds:
        print("  [FALTA] meta_entrenamiento.json no tiene detalle por fold "
              "(entrenaste con una versión anterior de train_model.py — vuelve a entrenar).")
        return

    print(f"  {'Fold':<10}{'MAE Codo (°)':<16}{'MAE Muñeca (°)':<18}{'R² promedio':<12}")
    for fr in folds:
        print(f"  {fr['fold']:<10}{fr['mae_codo']:<16.2f}{fr['mae_muneca']:<18.2f}{fr['r2_promedio']:<12.4f}")

    import statistics as st
    mae_c = [f["mae_codo"] for f in folds]
    mae_m = [f["mae_muneca"] for f in folds]
    r2p = [f["r2_promedio"] for f in folds]
    print(f"  {'Media±DE':<10}"
          f"{st.mean(mae_c):.2f}±{st.pstdev(mae_c):.2f}    "
          f"{st.mean(mae_m):.2f}±{st.pstdev(mae_m):.2f}      "
          f"{st.mean(r2p):.4f}±{st.pstdev(r2p):.4f}")


def tabla_6_8(meta):
    print("\n=== Tabla 6.8: Métricas sobre el conjunto de prueba ===")
    test = meta.get("regresor", {}).get("test")
    if not test:
        print("  [FALTA] No hay sección 'test' en meta_entrenamiento.json.")
        return
    print(f"  {'Articulación':<16}{'MAE (°)':<12}{'RMSE (°)':<12}{'R²':<10}")
    for col, m in test.items():
        rmse = m.get("test_rmse")
        rmse_str = f"{rmse:.2f}" if rmse is not None else "[FALTA re-entrenar]"
        print(f"  {col:<16}{m['test_mae']:<12.2f}{rmse_str:<12}{m['test_r2']:<10.4f}")


def tabla_6_9(meta):
    print("\n=== Tabla 6.9: Comparación entrenamiento vs. prueba (sobreajuste) ===")
    reg = meta.get("regresor", {})
    train, test, brecha = reg.get("train"), reg.get("test"), reg.get("brecha")
    if not (train and test and brecha):
        print("  [FALTA] meta_entrenamiento.json no tiene 'train'/'brecha' "
              "(entrenaste con una versión anterior de train_model.py — vuelve a entrenar).")
        return
    for col in test:
        print(f"  {col}:")
        print(f"    Entrenamiento  -> MAE={train[col]['train_mae']:.2f}°  R²={train[col]['train_r2']:.4f}")
        print(f"    Prueba         -> MAE={test[col]['test_mae']:.2f}°  R²={test[col]['test_r2']:.4f}")
        print(f"    Brecha         -> ΔMAE={brecha[col]['mae']:+.2f}°  ΔR²={brecha[col]['r2']:+.4f}")


def inferencia_pc(meta):
    print("\n=== Inferencia PC (columna de la Tabla 8.2) ===")
    inf = meta.get("regresor", {}).get("inferencia_pc")
    if not inf:
        print("  [FALTA] No hay medición de inferencia_pc (re-entrena con la versión actual).")
        return
    print(f"  Promedio: {inf['promedio_ms']:.2f} ms (±{inf['desv_est_ms']:.2f} ms)  "
          f"Máximo: {inf['maximo_ms']:.2f} ms  (n={inf['n_muestras']} muestras del set de prueba)")


def main():
    print("=" * 70)
    print("  REPORTE PARA LA TESIS — datos leídos desde archivos persistidos")
    print("=" * 70)

    tabla_6_6()

    meta = _cargar_json(RUTA_META)
    if meta is None:
        print("\n[FALTA] No existe models/meta_entrenamiento.json todavía.")
        print("-> Corre 'python training/train_model.py' al menos una vez.")
        sys.exit(0)

    tabla_6_7(meta)
    tabla_6_8(meta)
    tabla_6_9(meta)
    inferencia_pc(meta)
    print()


if __name__ == "__main__":
    main()
