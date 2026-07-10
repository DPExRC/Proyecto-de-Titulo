import csv

RUTA = "data/datos_emg.csv"

with open(RUTA, "r", newline="") as f:
    filas = list(csv.reader(f))

header_viejo = filas[0]
cuerpo = filas[1:]

filas_migradas = []
filas_ya_nuevas = []
for fila in cuerpo:
    if len(fila) == 14:
        # fila antigua, sin sesion_id/timestamp -> se rellenan como desconocidos
        filas_migradas.append(["legacy", ""] + fila)
    elif len(fila) == 16:
        filas_ya_nuevas.append(fila)
    else:
        print("Fila con largo inesperado, revisar a mano:", fila)

header_nuevo = ["sesion_id", "timestamp"] + header_viejo  # header_viejo ya son los 14 nombres

with open(RUTA, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header_nuevo)
    writer.writerows(filas_migradas + filas_ya_nuevas)