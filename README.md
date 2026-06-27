# Interfaz Míoeléctrica Híbrida para el Control de un Brazo Robótico (sEMG)

Este proyecto implementa un sistema de control mioeléctrico distribuido en tiempo real para un brazo robótico articulado mediante señales electromiográficas de superficie (sEMG). La arquitectura del sistema optimiza el uso de hardware mediante un esquema de **procesamiento descentralizado (offload)**, delegando las tareas matemáticas de alta carga computacional y la inferencia de Machine Learning a una PC, mientras que el microcontrolador actúa estrictamente como un puente de adquisición y pasarela de control mecánico.

---

## 🚀 Arquitectura General del Sistema

El flujo de información y procesamiento se divide en tres etapas claramente definidas:

```
[ ETAPA 1: ADQUISICIÓN (Hardware Embebido) ]
 Bíceps (A0) ------┐
 Tríceps (A1) ------├─→ Arduino Uno ──[ Transmisión Serial de Muestras Crudas ]
 Braquiorradial (A2)┘   (Puente DAQ)  

                               │
                               ▼ [ 115200 baud ]
                               │

[ ETAPA 2: DSP Y ML DUAL (PC - Módulo Python) ]
 ┌────────────────────────────────────────────────────────────────────────┐
 │ 1. FILTRADO DIGITAL (DSP)                                              │
 │    - Filtro IIR Butterworth Pasabanda (20–200 Hz, Orden 4, DF-I)       │
 ├────────────────────────────────────────────────────────────────────────┤
 │ 2. SEGMENTACIÓN Y EXTRACCIÓN DE FEATURES                               │
 │    - Buffer circular con ventanas deslizantes de 250 ms (Paso: 20 ms)  │
 │    - Cálculo de Características Temporales: [RMS, MAV, WL, ZCR]        │
 │    - Normalización Dinámica en base a calibración de sesión (%MVC)      │
 ├────────────────────────────────────────────────────────────────────────┤
 │ 3. PIPELINE DE INFERENCIA DUAL (Machine Learning)                      │
 │    - StandardScaler (Escalamiento de la matriz de entrada)             │
 │    - RandomForestClassifier ──► Predice Estado Macro (0, 1 o 2)        │
 │    - RandomForestRegressor  ──► Interpola Ángulo Continuo (0°–180°)    │
 └────────────────────────────────────────────────────────────────────────┘
                               │
                               ▼ [ Comandos de Ángulo Suavizados ]
                               │

[ ETAPA 3: CONTROL CINEMÁTICO ]
 PC Python ───→ Arduino Uno (Pasarela Serial-I2C) ───→ Driver PCA9685 ───→ Servos KS-3518

```

1. **Adquisición (Arduino Uno):** Configurado mediante interrupciones de hardware para muestrear de forma alternada 3 canales analógicos a una tasa consolidada de 1000 Hz. Envía las lecturas crudas del ADC sin procesar para evitar cuellos de botella por restricciones de SRAM.
2. **Procesamiento y Predicción (PC):** Módulo en Python que realiza el filtrado digital IIR, extrae características en ventanas móviles de 250 ms, normaliza los datos respecto a la Máxima Contracción Voluntaria (%MVC) y ejecuta un pipeline predictivo dual (Clasificación para estados macro + Regresión para interpolación angular continua).
3. **Control Mecánico:** La PC retorna las directrices angulares calculadas hacia el Arduino Uno, el cual despacha los setpoints por bus I2C al módulo PWM PCA9685 para mover los servomotores de alta torsión de la estructura robótica.

---

## 🛠️ Especificaciones de Hardware

| Componente | Conexión Física | Propósito |
| --- | --- | --- |
| Sensor sEMG Bíceps | Entrada Analógica **A0** | Captura de dinámica flexora del codo |
| Sensor sEMG Tríceps | Entrada Analógica **A1** | Captura de dinámica extensora del codo |
| Sensor sEMG Braquiorradial | Entrada Analógica **A2** | Captura de dinámicas de asistencia y rotación |
| Servomotores KS-3518 | Canales 0, 1 y 2 (PCA9685) | Actuadores de las articulaciones (0° - 180°) |
| Líneas de Control I2C | Pins **A4 (SDA)** y **A5 (SCL)** | Bus de comunicación hacia el driver PWM |
| Fuente de Poder Externa | Bornes V+ / GND del PCA9685 | Línea dedicada regulada a 5V/2A para evitar caídas de tensión |

---

## 📊 Pipeline Digital de Señales (DSP) y Machine Learning

### 1. Filtrado Digital

Para mitigar el ruido de baja frecuencia y evitar el solapamiento (*aliasing*) respetando el límite estricto de Nyquist por canal en la tasa de transferencia, se aplica un filtro digital **IIR Butterworth Pasabanda de 20 Hz a 200 Hz (Orden 4)** implementado en Direct Form I en Python.

### 2. Extracción de Características (*Features*)

Sobre ventanas dinámicas de 250 ms deslizantes cada 20 ms, se calculan de manera simultánea cuatro métricas en el dominio del tiempo por cada canal:

* **RMS** (*Root Mean Square*)
* **MAV** (*Mean Absolute Value*)
* **WL** (*Waveform Length*)
* **ZCR** (*Zero Crossing Rate*)

### 3. Inferencia Jerárquica Dual

* **RandomForestClassifier (200 árboles):** Determina el estado macro intencional del usuario en tres clases discretas: `REPOSO` (Clase 0), `FLEXIÓN` (Clase 1) o `EXTENSIÓN` (Clase 2).
* **RandomForestRegressor (200 árboles):** Si el clasificador detecta un estado activo (1 o 2), el regresor asume el control del lazo cinemático para interpolar de forma no lineal y fluida la trayectoria exacta del brazo en un rango continuo de `0° a 180°`.

---

## 📂 Estructura del Repositorio

```
├── firmware/
│   └── emg_v3/
│       └── emg_v3.ino         # Firmware de adquisición serie y pasarela I2C
├── src/
│   ├── config.py              # Parámetros globales y constantes (Fuente única de verdad)
│   ├── main.py                # Orquestador multihilo de producción en tiempo real
│   └── core/
│       └── serial_bridge.py   # Gestión del protocolo serial bidireccional
│   └── models/
│       └── predictor.py       # Interfaz de inferencia del pipeline unificado (RF)
├── data/
│   ├── captura.py             # Script automatizado de captura de señales etiquetadas
│   └── datos_emg.csv          # Dataset acumulado (Excluido del control de versiones)
├── training/
│   └── train.py               # Script de entrenamiento, validación cruzada y exportación
├── models/
│   ├── modelo_clasificador.pkl # Binario serializado del clasificador de estados macro
│   ├── modelo_regresor.pkl     # Binario serializado del regresor angular continuo
│   └── meta_entrenamiento.json # Reporte de métricas logradas (Accuracy, MAE, R²)
├── requirements.txt           # Dependencias de entorno del ecosistema Python
└── README.md                  # Documentación principal del sistema

```

---

## 🔧 Configuración e Instalación

### 1. Despliegue del Firmware

1. Conecta el Arduino Uno a tu computadora mediante el cable USB.
2. Abre el IDE de Arduino y carga el archivo ubicado en `firmware/emg_v3/emg_v3.ino`.
3. Instala la dependencia requerida desde el Gestor de Librerías: `Adafruit PWM Servo Driver Library`.
4. Sube (*Upload*) el programa al microcontrolador. Abre el Monitor Serie a **115200 baud** y verifica que imprima la cadena `READY`.

### 2. Configuración del Entorno Python

Asegúrate de contar con Python 3.8 o superior e instala los paquetes necesarios provistos en el archivo de requerimientos:

```bash
pip install -r requirements.txt

```

---

## 💻 Flujo de Trabajo Operacional

### Paso 1: Captura de Datos (*Dataset de Calibración*)

Para entrenar los modelos con tus propios patrones musculares, ejecuta el script de captura ejecutando rutinas para las distintas posiciones angulares fijadas para la calibración:

```bash
python data/captura.py --port COM3 --duracion 5 --rondas 5

```

*Este comando registrará ventanas sincronizadas de señales asociadas al ángulo objetivo y poblará de forma iterativa el archivo `data/datos_emg.csv`.*

### Paso 2: Entrenamiento del Pipeline Dual

Una vez construido el dataset, ejecuta el script de entrenamiento para validar y serializar los modelos inteligentes:

```bash
python training/train.py

```

*El script procesará la información, aplicará esquemas de validación cruzada (`StratifiedKFold` para clasificación y `KFold` para regresión), imprimirá matrices de confusión, el error absoluto medio (MAE) desglosado por rangos de movimiento y exportará los archivos `.pkl` resultantes en la carpeta `models/`.*

### Paso 3: Ejecución en Producción (Tiempo Real)

Para iniciar la operación en vivo del sistema distribuyendo las cargas de cómputo y controlando el brazo robótico de forma continua, ejecuta el orquestador principal:

```bash
python src/main.py

```

*(Nota: Si tu puerto serie difiere de `COM3`, puedes modificarlo directamente en `src/config.py`).*

---

## 🔒 Mecanismos de Robustez y Seguridad

* **Filtro de Histéresis Temporal:** Las decisiones del clasificador pasan por un algoritmo de votación por mayoría de 3 ciclos consecutivos, eliminando activaciones falsas o ruidos espurios transitorios (*chattering*).
* **Limitador de Tasa Cinemática:** El ángulo continuo arrojado por el regresor está acotado por software a un incremento máximo de `4.8° por ciclo` (basado en una velocidad angular real de los servos de $300^\circ/\text{s}$ en deltas de tiempo de $20\text{ ms}$), garantizando transiciones mecánicas fluidas y previniendo el desgaste de engranajes.

