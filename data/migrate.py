import csv
import os
import sys

RUTA = "data/datos_emg.csv"
LARGO_FILA_VIEJA = 14
LARGO_FILA_NUEVA = 16


def migrar(ruta: str = RUTA) -> bool:
    """Migra datos_emg.csv al esquema con sesion_id y timestamp.

    Idempotente: si el archivo ya tiene el encabezado nuevo, no modifica nada
    y devuelve False. Devuelve True si se escribió una migración."""
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No existe el archivo: {ruta}")

    with open(ruta, "r", newline="", encoding="utf-8") as f:
        filas = list(csv.reader(f))

    if not filas:
        raise ValueError(f"El archivo está vacío: {ruta}")

    header = filas[0]
    if header and header[0] == "sesion_id":
        return False

    cuerpo = filas[1:]
    filas_migradas = []
    filas_ya_nuevas = []

    for fila in cuerpo:
        if len(fila) == LARGO_FILA_VIEJA:
            filas_migradas.append(["legacy", ""] + fila)
        elif len(fila) == LARGO_FILA_NUEVA:
            filas_ya_nuevas.append(fila)
        else:
            raise ValueError(
                f"Fila con largo inesperado ({len(fila)} columnas): {fila}"
            )

    header_nuevo = ["sesion_id", "timestamp"] + header

    with open(ruta, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header_nuevo)
        writer.writerows(filas_migradas + filas_ya_nuevas)

    return True


def main() -> None:
    try:
        if migrar():
            print(f"Migración completada: {RUTA}")
        else:
            print(
                "El archivo ya tiene el esquema nuevo "
                "(sesion_id + timestamp). Nada que migrar."
            )
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
