# Proyecto EMG — Control de Servo por Señal Electromiográfica

**Versión:** 3.0

**Última actualización:** Junio 2026

**Músculos:** Bíceps (A0) / Tríceps (A1) / Antebrazo (A2)

**Clases Macro:** REPOSO / FLEXIÓN / EXTENSIÓN

---

## Arquitectura General

```
[ ETAPA 1: ADQUISICIÓN ]
Bíceps (A0) ----┐
Tríceps (A1) ----├─→ Arduino Uno (Puente DAQ - Muestras Crudas)
Antebrazo (A2) --┘
                       │
                       ▼ [ Transmisión Serial @ 115200 baud ]
                       │
[ ETAPA 2: DSP Y ML DUAL (PC - Python Offload) ]
 ┌────────────────────────────────────────────────────────┐
 │ 1. FILTRADO DIGITAL (DSP)                              │
 │    - Butterworth Pasabanda IIR (Orden 4) por canal     │
 ├────────────────────────────────────────────────────────┤
 │ 2. EXTRACCIÓN DE FEATURES (Ventana móvil 250 ms)       │
 │    - Vectores de Características: [RMS, MAV, WL, ZC]   │
 │    - Normalización dinámica %MVC por canal             │
 ├────────────────────────────────────────────────────────┤
 │ 3. PIPELINE DE INFERENCIA PIPELINE DUAL                │
 │    - RF Classifier ──► Determina Estado Macro          │
 │    - RF Regressor  ──► Calcula Ángulo Continuo (0-180°)│
 └────────────────────────────────────────────────────────┘
                       │
                       ▼ [ Comando de Posición Angular ]
                       │
[ ETAPA 3: CONTROL FÍSICO ]
 PC Python ──→ Arduino Uno (Pasarela Serial a I2C) ──→ Driver PCA9685 ──→ Servos KS-3518

```

El Arduino ejecuta únicamente la conversión analógica-digital (ADC) de los 3 canales de forma alternada y despacha los datos crudos hacia la PC. Python centraliza el procesamiento pesado (filtrado digital, extracción de características, normalización %MVC) y ejecuta el pipeline de Machine Learning dual. Posteriormente, la PC envía las directrices de posición angular calculadas al Arduino, el cual gestiona los actuadores a través del driver PCA9685 por interfaz I2C.

---

## Hardware

| Componente | Conexión | Descripción |
| --- | --- | --- |
| Módulo EMG bíceps (0–5V) | A0 | Canal analógico 0 |
| Módulo EMG tríceps (0–5V) | A1 | Canal analógico 1 |
| Módulo EMG antebrazo (0–5V) | A2 | Canal analógico 2 |
| PCA9685 SDA | A4 | Línea de datos del bus I2C |
| PCA9685 SCL | A5 | Línea de reloj del bus I2C |
| PCA9685 V+ | Fuente externa 5V/2A | Alimentación aislada para potencia de actuadores |
| PCA9685 GND | GND compartido con Arduino | Referencia común de señal |
| Servomotores KS-3518 | Canales 0, 1 y 2 del PCA9685 | Actuadores de articulaciones (Rango: 0° a 180°) |
| Oscilador PCA9685 | 25 MHz | Configurado para frecuencia base PWM de 50 Hz |

---

## Pipeline DSP (PC — Python Offload)

```
Muestras crudas recibidas por puerto serie (Fs = 1000 Hz total)
  │
  ├─ Canal 0 (bíceps)     ─→ IIR Butterworth Pasabanda 20–200 Hz, orden 4, Direct Form I
  ├─ Canal 1 (tríceps)    ─→ IIR Butterworth Pasabanda 20–200 Hz, orden 4, Direct Form I
  └─ Canal 2 (antebrazo)  ─→ IIR Butterworth Pasabanda 20–200 Hz, orden 4, Direct Form I
                                     │
                           Buffer circular 250 ms
                                     │
                           Cada 20 ms (Paso de ventana):
                             ├─ RMS (Root Mean Square) por canal
                             ├─ MAV (Mean Absolute Value) por canal
                             ├─ WL (Waveform Length) por canal
                             ├─ ZCR (Zero Crossing Rate) por canal
                             └─ Normalización %MVC dinámica por canal

```

> **Nota de Consistencia de Nyquist:** Para evitar fenómenos de aliasing, con una tasa de muestreo total de 1000 Hz distribuida de manera alternada entre canales, la frecuencia de corte superior del filtro Butterworth digital se ajusta a un máximo de 200 Hz para respetar el límite estricto de Nyquist por canal.

**Coeficientes del Filtro Digital IIR (Filtro Pasabanda):**

* **B (Numerador):** `[0.7320224766, 0.0, -1.4640449531, 0.0, 0.7320224766]`
* **A (Denominador):** `[1.0, -0.2627714585, -1.3636673385, 0.1365426158, 0.5371946248]`

---

## Pipeline ML Dual (PC — Python)

El vector de entrada consolidado integra las 4 características extraídas de todos los canales instrumentados, aplicando un escalamiento estandarizado antes de la ejecución en paralelo del ensamble:

```
Vector de Features X [rms, mav, wl, zcr] × canales
  │
  └─► StandardScaler
        │
        ├──► Modelo 1: RandomForestClassifier (200 árboles)
        │      │
        │      └─► Predicción de Estado Macro: 0 = REPOSO / 1 = FLEXIÓN / 2 = EXTENSIÓN
        │
        └──► Modelo 2: RandomForestRegressor (200 árboles)
               │
               └─► Inferencia de Posición Dinámica: Ángulo continuo en el rango de 0° a 180°

```

---

## Control de Servo e Inferencia Cinemática

La lógica de control combina el estado macro del clasificador con la interpolación fina calculada por el regresor para deparar una trayectoria continua en los servomotores:

* **Filtro de Histéresis:** Se implementa una ventana de votación por mayoría que requiere de 3 ciclos de control consecutivos para confirmar un cambio en el estado macro del clasificador, previniendo oscilaciones espurias (*chattering*).
* **Mapeo de Control Continuo:**
* Si la predicción macro es `REPOSO`, los servos convergen a su posición neutral de seguridad ($90^\circ$).
* Si la predicción es activa (`FLEXIÓN` o `EXTENSIÓN`), el control angular directo es gobernado en tiempo real por el ángulo continuo estimado por el regresor ($0^\circ\text{--}180^\circ$).


* **Limitación Cinemática:** Para garantizar suavidad y proteger los engranajes, se restringe el desplazamiento máximo por ciclo a `MAX_CAMBIO = 4.8°/ciclo` (derivado de una $\text{VEL\_MAX} = 300^\circ/\text{s}$ en un $\Delta t = 20\text{ ms}$).

---

## Protocolo Serial (115200 baud)

### Arduino → PC (Envío continuo de telemetría analógica cruda)

```
READY              Señal que indica el arranque exitoso del microcontrolador
D,<A0>,<A1>,<A2>\n  Valores enteros directos del ADC (rango 0–1023) por canal muestreado
# [...]            Líneas de depuración o comentarios (carácter inicial '#' ignorado por Python)

```

### PC → Arduino (Envío de comandos de control mecánico y configuración)

```
A,<ang0>,<ang1>,<ang2>\n   Comando directo de posición angular continua (0° a 180°) por articulación
c / C                     Iniciar procedimiento de calibración de umbrales en la sesión activa
m / M                     Conmutar entre modo de ejecución local autónomo / procesamiento remoto (offload)

```

---

## Calibración Dinámica (Gestionada en Python)

1. **Reposo (3-5s):** Determina el nivel de ruido base (`baseline_rms`) canal por canal en un estado de completa relajación muscular.
2. **MVC Máxima Contracción Voluntaria (3-5s por músculo):** Registra secuencialmente los picos de activación para el canal del Bíceps, Tríceps y Antebrazo de manera independiente durante esfuerzos máximos.

La normalización se aplica en ventanas deslizantes en tiempo real mediante la expresión:


$$\% \text{MVC}_{\text{ch}} = \frac{\text{rms}_{\text{ch}} - \text{baseline}_{\text{ch}}}{\text{mvc}_{\text{ch}} - \text{baseline}_{\text{ch}}} \times 100$$

---

## Estructura del Proyecto

```
firmware/
  emg_v3/
    emg_v3.ino         Firmware de adquisición y puente serie de datos crudos

src/
  config.py            Parámetros globales del sistema (fuente única de verdad)
  main.py              Orquestador multihilo de producción en PC (Muestreo + Inferencia)
  core/
    serial_bridge.py   Interfaz de comunicación bidireccional serial de alta velocidad
  models/
    predictor.py       Módulo encargado del pipeline dual (Clasificador + Regresor RF)

data/
  captura.py           Script de captura y registro indexado de datos biomédicos
  datos_emg.csv        [generado — excluido del sistema de control de versiones]

training/
  train.py             Script de entrenamiento dual (Evaluación cruzada K-Fold + exportación)

models/
  modelo_clasificador.pkl   [generado — serialización del clasificador de estados]
  modelo_regresor.pkl       [generado — serialización del regresor continuo de ángulos]
  meta_entrenamiento.json   [generado — almacenamiento de métricas MAE, Accuracy y R²]

requirements.txt

```

---

## Flujo de Uso

### 1. Flashear firmware

```
Arduino IDE → firmware/emg_v3/emg_v3.ino → Upload
Verificar en el monitor serial (115200 baud) el mensaje de inicialización: READY

```

### 2. Instalar dependencias Python

```bash
pip install -r requirements.txt

```

### 3. Capturar dataset de entrenamiento

```bash
python data/captura.py --port COM3 --duracion 5 --rondas 5

```

* Automatiza la toma sincronizada asociando características biomédicas a los ángulos fijos planteados para la calibración. El archivo `data/datos_emg.csv` añade los registros iterativamente.

### 4. Entrenar el pipeline dual

```bash
python training/train.py

```

* Computa el entrenamiento del clasificador (`StratifiedKFold`) y del regresor (`KFold`). Genera los reportes de exactitud, matrices de confusión correspondientes y el error absoluto medio (MAE) desglosado por rangos angulares críticos. Exporta los binarios serializados a la carpeta `models/`.

### 5. Ejecución en tiempo real

```bash
python src/main.py

```

---

## Estado del Proyecto

### Implementado

| Componente | Detalle |
| --- | --- |
| Adquisición Remota | Firmware optimizado para muestreo alternado a 1000 Hz total en los canales analógicos |
| Arquitectura Offload | Pipeline matemático y de filtrado digital IIR Butterworth portado íntegramente a PC en Python |
| Extracción de Features | Cálculo simultáneo en ventanas de 250 ms para métricas temporales: RMS, MAV, WL y ZCR |
| Pipeline ML Dual | Implementación unificada de RandomForestClassifier y RandomForestRegressor operando en paralelo |
| Robustez de Control | Mecanismos de histéresis temporal por mayoría y limitador cinemático de tasa instalados |
| Comunicación Bidireccional | Protocolo unificado serial que opera estable a una velocidad fija de 115200 baud |

### Pendiente

| Tarea | Prioridad |
| --- | --- |
| Captura de dataset exhaustivo con sujeto real y validación de las curvas del regresor | Alta |
| Pruebas empíricas de usabilidad del lazo de control con escala estandarizada SUS | Alta |
| Análisis experimental cuantitativo de retardos y latencias en el canal serie | Media |
| Persistencia estática de los parámetros de calibración (%MVC) mediante archivos estructurados JSON | Baja |

---

## Dependencias

### Arduino

```cpp
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

```

### Python

```
numpy>=1.21
scipy>=1.7
pandas>=1.3
scikit-learn>=1.2
joblib>=1.1
pyserial>=3.5

```
