<!-- markdownlint-disable-file MD046 -->
# Sistema EMG para control de brazo robótico

Este proyecto implementa un flujo completo de adquisición, procesamiento y control en tiempo real de señales electromiográficas de superficie (sEMG) para mover un brazo robótico mediante dos grados de libertad: codo y muñeca.

La arquitectura actual está organizada en tres capas:

1. Firmware embebido en Arduino para adquisición y transmisión de muestras.
2. Procesamiento digital en Python para filtrado, extracción de características y normalización.
3. Inferencia con un modelo de regresión basado en RandomForest para producir ángulos de referencia.

---

## ✅ Estado actual del proyecto

La versión actual del repositorio incluye:

- Captura de datos EMG desde Arduino mediante puerto serial.
- Extracción de características temporales RMS, MAV, WL y ZCR por canal.
- Calibración baseline/MVC por sesión.
- Normalización offline a porcentaje de MVC.
- Entrenamiento de un modelo de regresión multi-salida para codo y muñeca.
- Modo interactivo principal en Python para entrenar o usar el sistema.

---

## 🔧 Arquitectura

```text
[Hardware]
Arduino Uno ──> Lecturas analógicas (3 canales sEMG)

[Procesamiento en PC]
Python ──> Filtrado + ventanas + features + normalización %MVC ──> modelo

[Control]
Python ──> comandos angulares ──> Arduino ──> servomotores
```

El flujo actual usa estas señales y características:

- 3 canales sEMG: bíceps, tríceps y antebrazo.
- 12 features: RMS, MAV, WL y ZCR por canal.
- 2 DOF controlados: codo y muñeca.
- Modelo: RandomForestRegressor multi-salida.

---

## 🧰 Requisitos

- Python 3.9 o superior.
- Arduino con firmware compatible.
- Dependencias Python:

```bash
pip install -r requirements.txt
pip install -e .
```

La instalación editable registra el paquete `emg_arm` y permite ejecutar
`training/train_model.py` directamente y correr `pytest` sin flags adicionales.

---

## 📦 Instalación y uso

### 1. Subir el firmware

Carga el archivo de firmware en el Arduino desde:

- [firmware/emg_bridge_v4.ino](firmware/emg_bridge_v4.ino)

Asegúrate de abrir el monitor serie a 115200 baudios y verificar que el firmware responda con READY.

### 2. Ejecutar la interfaz principal

Desde la raíz del proyecto:

```bash
python main.py
```

El menú ofrece dos modos:

- Entrenar: calibra, captura datos, normaliza y entrena el modelo.
- Usar: carga la calibración y el modelo entrenado para inferencia en tiempo real.

### 3. Flujo recomendado

#### Captura de datos

```bash
python data/capture.py --port COM5 --duracion 5
```

Esto genera un archivo CSV con características crudas en:

- [data/datos_emg.csv](data/datos_emg.csv)

#### Normalización %MVC

```bash
python emg_arm/processing/standardization.py
```

Esto produce:

- [data/datos_emg_normalizado.csv](data/datos_emg_normalizado.csv)

#### Entrenamiento del modelo

```bash
python training/train_model.py
```

El modelo entrenado se guarda en:

- [models/modelo_regresor.pkl](models/modelo_regresor.pkl)
- [models/meta_entrenamiento.json](models/meta_entrenamiento.json)

---

## 📁 Estructura del repositorio

```text
Servos/
├── data/
│   ├── capture.py
│   ├── calibracion.json
│   ├── datos_emg.csv
│   └── datos_emg_normalizado.csv
├── firmware/
│   └── emg_bridge_v4.ino
├── models/
│   └── meta_entrenamiento.json
├── src/
│   ├── config.py
│   ├── core/
│   │   └── serial_bridge.py
│   ├── models/
│   │   └── predictor.py
│   └── processing/
│       ├── calibration.py
│       ├── dsp.py
│       ├── features.py
│       ├── filter.py
│       └── standardization.py
├── training/
│   └── train_model.py
├── main.py
├── requirements.txt
└── README.md
```

---

## 📝 Notas importantes

- El puerto serial por defecto está definido en [src/config.py](src/config.py).
- Si cambias la forma de capturar datos o la calibración, conviene recalcular la normalización antes de entrenar.
- El sistema está pensado para funcionar en sesiones de calibración y uso consistentes, ya que la normalización depende de la calibración registrada.
