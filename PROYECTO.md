# Proyecto EMG — Control de Servo por Señal Electromiográfica

**Versión:** 3.0  
**Última actualización:** Junio 2026  
**Músculos:** Bíceps (A0) / Tríceps (A1)  
**Clases:** REPOSO / FLEXIÓN / EXTENSIÓN  

---

## Arquitectura General

```
Bíceps (A0) --──┐
                ├─→ Arduino Uno (ADC + IIR + RMS + ZCR) ──→ PC Python (RF) ──→ Arduino (servo)
Tríceps (A1) --─┘
                |
Antebrazo (A2) ─┘
```

El Arduino ejecuta íntegramente el DSP (filtrado, RMS, ZCR, normalización %MVC)
y transmite vectores de features al PC. Python clasifica y devuelve la clase predicha.
El Arduino aplica el limitador de tasa y mueve el servo vía PCA9685.

---

## Hardware

| Componente | Conexión |
|---|---|
| Módulo EMG bíceps (0–5V) | A0 |
| Módulo EMG tríceps (0–5V) | A1 |
| PCA9685 SDA | A4 |
| PCA9685 SCL | A5 |
| PCA9685 V+ | Fuente externa 5V/2A |
| PCA9685 GND | GND compartido con Arduino |
| Servo KS-3518 | Canal 0 del PCA9685 |
| Oscilador PCA9685 | 25 MHz (configurado por firmware) |

---

## Pipeline DSP (embebido en Arduino)

```
ADC alternado A0/A1 — Timer1 a 1000 Hz total (500 Hz por canal)
  │
  ├─ Canal 0 (bíceps)  ─→ IIR Butterworth 20–450 Hz, orden 4, Direct Form I
  └─ Canal 1 (tríceps) ─→ IIR Butterworth 20–450 Hz, orden 4, Direct Form I
                                   │
                          Buffer circular 250 ms
                                   │
                          Cada 20 ms (N_PASO = 10 muestras/canal):
                            ├─ RMS por canal
                            ├─ ZCR por canal
                            └─ Normalización %MVC
                                       │
                              TX: F,<rms0>,<zcr0>,<rms1>,<zcr1>\n
```

**Coeficientes IIR:**
- B = [0.7320224766, 0, −1.4640449531, 0, 0.7320224766]
- A = [1.0, −0.2627714585, −1.3636673385, 0.1365426158, 0.5371946248]

---

## Pipeline ML (PC — Python)

```
Vector F [rms0, zcr0, rms1, zcr1]
  │
  └─ StandardScaler → RandomForestClassifier (200 árboles)
          │
          └─ Clase: 0=REPOSO / 1=FLEXION / 2=EXTENSION
                  │
             TX: C,<clase>\n → Arduino
```

---

## Control de Servo

| Clase | Ángulo destino |
|---|---|
| REPOSO | 90° |
| FLEXIÓN | 170° |
| EXTENSIÓN | 10° |

Limitador de tasa: MAX\_CAMBIO = 4.8°/ciclo (VEL\_MAX=300°/s × DT=20ms × 0.8)  
Histéresis de clase: confirmación en 3 ciclos consecutivos antes de cambiar estado.

---

## Protocolo Serial (115200 baud)

### Arduino → PC
```
READY                              arranque del firmware
F,<rms0>,<zcr0>,<rms1>,<zcr1>     vector de features normalizado (%MVC y ZCR)
# [...]                            líneas de log/diagnóstico (ignoradas por Python)
```

### PC → Arduino
```
C,<0|1|2>\n    clase predicha (0=REPOSO, 1=FLEXION, 2=EXTENSION)
A,<angulo>\n   control directo de ángulo (opcional)
c / C          recalibrar
m / M          alternar modo local / PC offload
```

---

## Calibración (3 fases, embebida en Arduino)

1. **Reposo (3s):** baseline\_rms por canal (media de ventanas RMS)
2. **MVC bíceps (3s):** máximo RMS de bíceps con tríceps relajado
3. **MVC tríceps (3s):** máximo RMS de tríceps con bíceps relajado

Normalización %MVC en cada ciclo:
```
%MVC_ch = (rms_ch - baseline_ch) / (mvc_ch - baseline_ch) × 100
```

---

## Estructura del Proyecto

```
firmware/
  emg_v3/
    emg_v3.ino          Firmware principal v3.0

src/
  config.py             Parámetros globales (fuente única de verdad)
  main.py               Orquestador de producción (2 hilos)
  core/
    serial_bridge.py    Interfaz serial con el firmware
  models/
    predictor.py        Clasificador RF (con fallback por umbral)

data/
  captura.py            Captura de dataset etiquetado
  datos_emg.csv         [generado — excluir del repositorio]

training/
  train.py              Entrenamiento RF + evaluación + exportación

models/
  modelo_emg.pkl        [generado — excluir del repositorio]
  modelo_emg_meta.json  [generado — metadatos del entrenamiento]

requirements.txt
```

---

## Flujo de Uso

### 1. Flashear firmware

```
Arduino IDE → firmware/emg_v3/emg_v3.ino → Upload
Monitor serial (115200 baud): debe aparecer READY
```

### 2. Instalar dependencias Python

```bash
pip install -r requirements.txt
```

### 3. Capturar dataset

```bash
python data/captura.py --port COM3 --duracion 5 --rondas 5
```

- 3 clases × 5 rondas × 5s × ~50 vectores/s ≈ 750 vectores por clase
- El firmware debe estar en modo PC offload (post-calibración automática)
- El archivo `data/datos_emg.csv` se acumula en ejecuciones sucesivas

### 4. Entrenar modelo

```bash
python training/train.py
```

- Reporta accuracy, balanced accuracy, matriz de confusión e importancia de features
- Si balanced accuracy < 0.80: capturar más datos
- Genera `models/modelo_emg.pkl` y `models/modelo_emg_meta.json`

### 5. Ejecutar en producción

```bash
python src/main.py
```

Cambiar `PORT` en `src/config.py` si el puerto es distinto a `COM3`.

---

## Estado del Proyecto

### Implementado

| Componente | Detalle |
|---|---|
| Firmware v3.0 | DAQ dual canal bíceps/tríceps, DSP embebido completo |
| Muestreo | Timer1 alternado, 500 Hz/canal (1000 Hz total) |
| Filtro IIR | Butterworth 20–450 Hz orden 4, Direct Form I, ambos canales |
| Features | RMS + ZCR por canal, ventana 250 ms / paso 20 ms |
| Normalización | %MVC con calibración de 3 fases por canal |
| Protocolo serial | F,... (TX) / C,... (RX) / líneas # para log |
| Histéresis de clase | 3 ciclos de confirmación |
| Limitador de tasa | MAX_CAMBIO = 4.8°/ciclo |
| Pipeline Python | config / serial\_bridge / predictor / main / captura / train |
| Clasificador | RandomForest + StandardScaler, validación cruzada 5-fold |

### Pendiente

| Tarea | Prioridad |
|---|---|
| Captura de dataset con sujeto real y evaluación de balanced accuracy | Alta |
| Pruebas de usabilidad (instrumento SUS) — Espiral 5 | Alta |
| Métricas cuantitativas de tiempo de subida/bajada (de serial Angulo) | Media |
| Persistencia de calibración (JSON) entre sesiones | Baja |

---

## Dependencias

### Arduino
```cpp
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
// Instalar: Adafruit PWM Servo Driver Library (Library Manager)
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
