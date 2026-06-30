// =============================================================================
// EMG v3.0 — DAQ DUAL CANAL + DSP EMBEBIDO + PC OFFLOAD
// =============================================================================
// Hardware:   Arduino Uno (ATmega328P, 16 MHz)
//             Módulo EMG genérico x2: bíceps → A0, tríceps → A1
//             PCA9685 (I2C, 25 MHz): SDA=A4, SCL=A5
//             Servo KS-3518: canal 0 del PCA9685
// Fs:         1000 Hz por canal (muestreo alternado A0/A1)
// DSP:        Butterworth pasabanda orden 4 (20–450 Hz), Direct Form I
// Features:   RMS + ZCR por canal — ventana 250 ms, paso 20 ms
// Protocolo TX (Arduino → PC):  F,<rms0>,<zcr0>,<rms1>,<zcr1>\n
// Protocolo RX (PC → Arduino):  C,<clase>\n  (0=REPOSO, 1=FLEXION, 2=EXTENSION)
//                                A,<angulo>\n (opcional, control directo)
// Calibración: baseline reposo + MVC por canal, bajo demanda ('c'/'C')
// Serial:     115200 baud
// =============================================================================

#include <avr/interrupt.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// ---------------------------------------------------------------------------
// PCA9685
// ---------------------------------------------------------------------------
Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(0x40);

#define CANAL_SERVO      0
#define PULSO_MIN_US     500
#define PULSO_MAX_US     2400
#define FRECUENCIA_PWM   50

#define US_POR_CUENTA    (20000.0f / 4096.0f)
#define PULSO_MIN_CNT    ((uint16_t)(PULSO_MIN_US / US_POR_CUENTA))
#define PULSO_MAX_CNT    ((uint16_t)(PULSO_MAX_US / US_POR_CUENTA))

// ---------------------------------------------------------------------------
// PINES EMG
// ---------------------------------------------------------------------------
#define PIN_BICEPS       A0
#define PIN_TRICEPS      A1

// ---------------------------------------------------------------------------
// PARÁMETROS DE MUESTREO Y VENTANA
// ---------------------------------------------------------------------------
#define FS               1000      // Hz por canal
#define VENTANA_MS       250
#define PASO_MS          20
#define N_VENTANA        250       // muestras por canal por ventana
#define N_PASO           20        // muestras por canal por paso

// ---------------------------------------------------------------------------
// COEFICIENTES IIR — Butterworth pasabanda orden 4, 20–450 Hz, Fs=1000 Hz
// Direct Form I
// ---------------------------------------------------------------------------
const float B_COEF[5] = {
   0.7320224766f,
   0.0f,
  -1.4640449531f,
   0.0f,
   0.7320224766f
};
const float A_COEF[5] = {
   1.0f,
  -0.2627714585f,
  -1.3636673385f,
   0.1365426158f,
   0.5371946248f
};

// ---------------------------------------------------------------------------
// PARÁMETROS DE CONTROL
// ---------------------------------------------------------------------------
// Ángulos de destino por clase
#define ANGULO_REPOSO    90.0f    // posición neutra
#define ANGULO_FLEXION   170.0f   // bíceps contrae → flexión
#define ANGULO_EXTENSION 10.0f    // tríceps contrae → extensión

// Limitador de tasa de variación (MAX_CAMBIO = VEL_MAX × DT × margen)
// KS-3518: 300°/s sin carga — margen 80%
#define VEL_MAX_SERVO    300.0f
#define DT_CONTROL       (PASO_MS / 1000.0f)
#define MAX_CAMBIO       (VEL_MAX_SERVO * DT_CONTROL * 0.8f)  // 4.8°/ciclo

// Histéresis de clase para evitar oscilación entre estados
#define MIN_CICLOS_CLASE 3

// ---------------------------------------------------------------------------
// BUFFERS Y ESTADO — Canal 0 (bíceps) y Canal 1 (tríceps)
// ---------------------------------------------------------------------------
float    buf_ch0[N_VENTANA];
float    buf_ch1[N_VENTANA];
uint16_t idx_buf        = 0;
uint32_t contador       = 0;

// Historiales IIR por canal
float x0_hist[4] = {0}; float y0_hist[4] = {0};  // bíceps
float x1_hist[4] = {0}; float y1_hist[4] = {0};  // tríceps

// Muestras crudas (ISR alterna entre canales)
volatile int16_t muestra_ch0 = 0;
volatile int16_t muestra_ch1 = 0;
volatile bool    hay_muestra  = false;
volatile uint8_t canal_actual = 0;   // 0=bíceps, 1=tríceps

// Calibración por canal
float baseline_ch0 = 0.0f; float mvc_ch0 = 512.0f;
float baseline_ch1 = 0.0f; float mvc_ch1 = 512.0f;
bool  calibrado    = false;

// Control servo
float angulo_actual  = ANGULO_REPOSO;
float meta_angulo    = ANGULO_REPOSO;
uint8_t clase_actual = 0;            // 0=reposo, 1=flexion, 2=extension
uint8_t ciclos_clase = 0;            // histéresis

// Modo de operación
// 0 = control local por umbral (sin PC), 1 = PC offload activo
uint8_t modo = 0;

// ---------------------------------------------------------------------------
// TIMER1 — muestreo alternado a 500 Hz efectivo por canal (1000 Hz total)
// Cada disparo lee un canal y alterna al siguiente
// ---------------------------------------------------------------------------
void configurarTimer1() {
  cli();
  TCCR1A = 0; TCCR1B = 0; TCNT1 = 0;
  OCR1A  = 1999;                          // 1 ms — 1000 Hz total
  TCCR1B |= (1 << WGM12);
  TCCR1B |= (1 << CS11);                  // prescaler 8
  TIMSK1 |= (1 << OCIE1A);
  sei();
}

ISR(TIMER1_COMPA_vect) {
  if (canal_actual == 0) {
    muestra_ch0 = analogRead(PIN_BICEPS);
    canal_actual = 1;
  } else {
    muestra_ch1 = analogRead(PIN_TRICEPS);
    canal_actual = 0;
    hay_muestra  = true;   // par completo disponible cada 2 ms (500 Hz/canal)
  }
}

// ---------------------------------------------------------------------------
// FILTRO IIR — Direct Form I, orden 4
// ---------------------------------------------------------------------------
float filtrarIIR(float x0, float *xh, float *yh) {
  float y0 = B_COEF[0]*x0  + B_COEF[1]*xh[0] + B_COEF[2]*xh[1]
           + B_COEF[3]*xh[2] + B_COEF[4]*xh[3]
           - A_COEF[1]*yh[0] - A_COEF[2]*yh[1]
           - A_COEF[3]*yh[2] - A_COEF[4]*yh[3];
  xh[3]=xh[2]; xh[2]=xh[1]; xh[1]=xh[0]; xh[0]=x0;
  yh[3]=yh[2]; yh[2]=yh[1]; yh[1]=yh[0]; yh[0]=y0;
  return y0;
}

// ---------------------------------------------------------------------------
// RMS — sobre el buffer circular completo
// ---------------------------------------------------------------------------
float calcularRMS(float *buf) {
  float suma = 0.0f;
  for (uint16_t i = 0; i < N_VENTANA; i++)
    suma += buf[i] * buf[i];
  return sqrt(suma / N_VENTANA);
}

// ---------------------------------------------------------------------------
// ZCR — conteo de cruces por cero normalizados por N_VENTANA
// ---------------------------------------------------------------------------
float calcularZCR(float *buf) {
  uint16_t cruces = 0;
  for (uint16_t i = 1; i < N_VENTANA; i++) {
    if ((buf[i] >= 0.0f && buf[i-1] < 0.0f) ||
        (buf[i] <  0.0f && buf[i-1] >= 0.0f))
      cruces++;
  }
  return (float)cruces / (float)(N_VENTANA - 1);
}

// ---------------------------------------------------------------------------
// SERVO
// ---------------------------------------------------------------------------
void moverServo(float angulo) {
  angulo = constrain(angulo, 0.0f, 180.0f);
  uint16_t cuentas = (uint16_t)map(
    (long)(angulo * 10), 0, 1800, PULSO_MIN_CNT, PULSO_MAX_CNT
  );
  pca.setPWM(CANAL_SERVO, 0, cuentas);
}

// Aplica limitador de tasa y mueve el servo
void actualizarServo() {
  float dif = meta_angulo - angulo_actual;
  if (abs(dif) > MAX_CAMBIO)
    angulo_actual += (dif > 0 ? 1.0f : -1.0f) * MAX_CAMBIO;
  else
    angulo_actual = meta_angulo;
  angulo_actual = constrain(angulo_actual, 0.0f, 180.0f);
  moverServo(angulo_actual);
}

// Traduce clase a ángulo destino
void aplicarClase(uint8_t clase) {
  switch (clase) {
    case 1:  meta_angulo = ANGULO_FLEXION;   break;
    case 2:  meta_angulo = ANGULO_EXTENSION; break;
    default: meta_angulo = ANGULO_REPOSO;    break;
  }
}

// ---------------------------------------------------------------------------
// CALIBRACIÓN
// ---------------------------------------------------------------------------
void contarRegresivo(uint8_t s) {
  for (; s > 0; s--) {
    Serial.print(F("# → ")); Serial.print(s); Serial.println(F("s..."));
    delay(1000);
  }
}

// Recolecta RMS acumulado de ambos canales durante duracion_ms
void recolectar(uint32_t duracion_ms,
                float &suma0, uint16_t &n0, float &max0,
                float &suma1, float &max1) {
  suma0=0; n0=0; max0=0; suma1=0; max1=0;
  uint32_t t0 = millis();
  while (millis() - t0 < duracion_ms) {
    if (hay_muestra) {
      hay_muestra = false;
      float f0 = filtrarIIR((float)muestra_ch0, x0_hist, y0_hist);
      float f1 = filtrarIIR((float)muestra_ch1, x1_hist, y1_hist);
      buf_ch0[idx_buf % N_VENTANA] = f0;
      buf_ch1[idx_buf % N_VENTANA] = f1;
      idx_buf++; contador++;
      if (contador >= N_VENTANA && (contador % N_PASO == 0)) {
        float r0 = calcularRMS(buf_ch0);
        float r1 = calcularRMS(buf_ch1);
        suma0 += r0; n0++;
        if (r0 > max0) max0 = r0;
        if (r1 > max1) max1 = r1;
        suma1 += r1;
      }
    }
  }
}

void calibrar() {
  calibrado = false;
  moverServo(ANGULO_REPOSO);
  memset(buf_ch0,  0, sizeof(buf_ch0));
  memset(buf_ch1,  0, sizeof(buf_ch1));
  memset(x0_hist, 0, sizeof(x0_hist));
  memset(y0_hist, 0, sizeof(y0_hist));
  memset(x1_hist, 0, sizeof(x1_hist));
  memset(y1_hist, 0, sizeof(y1_hist));
  idx_buf=0; contador=0; angulo_actual=ANGULO_REPOSO; meta_angulo=ANGULO_REPOSO;

  Serial.println(F("#"));
  Serial.println(F("# ╔══════════════════════════════════════╗"));
  Serial.println(F("# ║         CALIBRACIÓN EMG v3.0         ║"));
  Serial.println(F("# ╚══════════════════════════════════════╝"));
  Serial.println(F("# Canales: A0=Bíceps  A1=Tríceps"));
  Serial.println(F("#"));

  float s0,s1,m0,m1; uint16_t n0;

  // FASE 1 — REPOSO
  Serial.println(F("# FASE 1/3: REPOSO — relaja ambos músculos"));
  contarRegresivo(3);
  recolectar(3000, s0, n0, m0, s1, m1);
  baseline_ch0 = (n0 > 0) ? s0/n0 : 0.0f;
  baseline_ch1 = (n0 > 0) ? s1/n0 : 0.0f;
  Serial.print(F("# Baseline bíceps="));  Serial.print(baseline_ch0, 4);
  Serial.print(F("  tríceps="));          Serial.println(baseline_ch1, 4);

  memset(buf_ch0,0,sizeof(buf_ch0)); memset(buf_ch1,0,sizeof(buf_ch1));
  idx_buf=0; contador=0;

  // FASE 2 — MVC BÍCEPS
  Serial.println(F("# FASE 2/3: MVC BÍCEPS — contrae bíceps al máximo, relaja tríceps"));
  contarRegresivo(3);
  recolectar(3000, s0, n0, m0, s1, m1);
  mvc_ch0 = (m0 > baseline_ch0 && m0 >= 1.0f) ? m0 : 512.0f;
  Serial.print(F("# MVC bíceps="));       Serial.println(mvc_ch0, 4);

  memset(buf_ch0,0,sizeof(buf_ch0)); memset(buf_ch1,0,sizeof(buf_ch1));
  idx_buf=0; contador=0;

  // FASE 3 — MVC TRÍCEPS
  Serial.println(F("# FASE 3/3: MVC TRÍCEPS — contrae tríceps al máximo, relaja bíceps"));
  contarRegresivo(3);
  recolectar(3000, s0, n0, m0, s1, m1);
  mvc_ch1 = (m1 > baseline_ch1 && m1 >= 1.0f) ? m1 : 512.0f;
  Serial.print(F("# MVC tríceps="));      Serial.println(mvc_ch1, 4);

  // Validación mínima
  if (mvc_ch0 == 512.0f)
    Serial.println(F("# ADVERTENCIA: bíceps sin contracción válida — usando rango por defecto"));
  if (mvc_ch1 == 512.0f)
    Serial.println(F("# ADVERTENCIA: tríceps sin contracción válida — usando rango por defecto"));

  // Reset final
  memset(buf_ch0,0,sizeof(buf_ch0)); memset(buf_ch1,0,sizeof(buf_ch1));
  memset(x0_hist,0,sizeof(x0_hist)); memset(y0_hist,0,sizeof(y0_hist));
  memset(x1_hist,0,sizeof(x1_hist)); memset(y1_hist,0,sizeof(y1_hist));
  idx_buf=0; contador=0;
  calibrado = true;

  Serial.println(F("# ╔══════════════════════════════════════╗"));
  Serial.println(F("# ║      CALIBRACIÓN COMPLETADA ✓        ║"));
  Serial.println(F("# ╚══════════════════════════════════════╝"));
  Serial.println(F("# Modo PC offload activo. Envía 'c' para recalibrar, 'm' para modo local."));
  Serial.println(F("#"));
  modo = 1;   // activa PC offload tras calibración
  delay(500);
}

// ---------------------------------------------------------------------------
// PARSER DE COMANDOS SERIALES DESDE PC
// ---------------------------------------------------------------------------
// Buffer de recepción
char rx_buf[16];
uint8_t rx_idx = 0;

void procesarLinea(char *linea) {
  if (linea[0] == 'C' && linea[1] == ',') {
    // C,<clase>
    uint8_t clase_nueva = (uint8_t)atoi(linea + 2);
    if (clase_nueva <= 2) {
      // Histéresis: confirmar MIN_CICLOS_CLASE antes de cambiar
      if (clase_nueva == clase_actual) {
        ciclos_clase++;
      } else {
        ciclos_clase = 1;
        clase_actual = clase_nueva;
      }
      if (ciclos_clase >= MIN_CICLOS_CLASE) {
        aplicarClase(clase_actual);
      }
    }
  } else if (linea[0] == 'A' && linea[1] == ',') {
    // A,<angulo> — control directo de ángulo
    float ang = atof(linea + 2);
    meta_angulo = constrain(ang, 0.0f, 180.0f);
  }
}

void leerSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == 'c' || c == 'C' && rx_idx == 0) {
      calibrar();
      return;
    }
    if (c == 'm' || c == 'M') {
      modo = (modo == 0) ? 1 : 0;
      Serial.print(F("# Modo: "));
      Serial.println(modo == 1 ? F("PC offload") : F("local"));
      return;
    }
    if (c == '\n') {
      rx_buf[rx_idx] = '\0';
      rx_idx = 0;
      procesarLinea(rx_buf);
    } else if (rx_idx < 15) {
      rx_buf[rx_idx++] = c;
    }
  }
}

// ---------------------------------------------------------------------------
// SETUP
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  Wire.begin();
  pca.begin();
  pca.setOscillatorFrequency(25000000);
  pca.setPWMFreq(FRECUENCIA_PWM);
  delay(10);
  moverServo(ANGULO_REPOSO);
  pinMode(PIN_BICEPS,  INPUT);
  pinMode(PIN_TRICEPS, INPUT);
  configurarTimer1();

  Serial.println(F("READY"));
  Serial.println(F("# EMG v3.0 — DAQ dual canal bíceps/tríceps"));
  Serial.println(F("# Fs=500Hz/canal | Ventana=250ms | Paso=20ms"));
  Serial.println(F("# Protocolo TX: F,rms0,zcr0,rms1,zcr1"));
  Serial.println(F("# Protocolo RX: C,<0|1|2>  o  A,<angulo>"));
  Serial.println(F("# 'c' = recalibrar | 'm' = alternar modo local/offload"));
  Serial.println(F("#"));
  delay(1000);
  calibrar();
}

// ---------------------------------------------------------------------------
// LOOP PRINCIPAL
// ---------------------------------------------------------------------------
void loop() {
  leerSerial();

  if (!hay_muestra || !calibrado) return;
  hay_muestra = false;

  // 1. Filtrar ambos canales
  float f0 = filtrarIIR((float)muestra_ch0, x0_hist, y0_hist);
  float f1 = filtrarIIR((float)muestra_ch1, x1_hist, y1_hist);

  // 2. Buffer circular
  uint16_t pos = idx_buf % N_VENTANA;
  buf_ch0[pos] = f0;
  buf_ch1[pos] = f1;
  idx_buf++;
  contador++;

  // 3. Cada N_PASO muestras (20 ms), calcular features y actuar
  if (contador >= N_VENTANA && (contador % N_PASO == 0)) {

    float rms0 = calcularRMS(buf_ch0);
    float rms1 = calcularRMS(buf_ch1);
    float zcr0 = calcularZCR(buf_ch0);
    float zcr1 = calcularZCR(buf_ch1);

    // Normalización %MVC
    float mvc_pct0 = constrain(
      (rms0 - baseline_ch0) / (mvc_ch0 - baseline_ch0) * 100.0f, 0.0f, 100.0f);
    float mvc_pct1 = constrain(
      (rms1 - baseline_ch1) / (mvc_ch1 - baseline_ch1) * 100.0f, 0.0f, 100.0f);

    if (modo == 1) {
      // ── MODO PC OFFLOAD: transmitir features normalizadas ────────────────
      // Formato: F,<rms0_%mvc>,<zcr0>,<rms1_%mvc>,<zcr1>
      Serial.print(F("F,"));
      Serial.print(mvc_pct0, 3); Serial.print(F(","));
      Serial.print(zcr0, 5);     Serial.print(F(","));
      Serial.print(mvc_pct1, 3); Serial.print(F(","));
      Serial.println(zcr1, 5);

    } else {
      // ── MODO LOCAL: clasificación por umbral simple ─────────────────────
      // Usado durante capturas de dataset o si el PC no está conectado
      uint8_t clase_local;
      if (mvc_pct0 > 30.0f && mvc_pct1 < 20.0f)
        clase_local = 1;   // FLEXION
      else if (mvc_pct1 > 30.0f && mvc_pct0 < 20.0f)
        clase_local = 2;   // EXTENSION
      else
        clase_local = 0;   // REPOSO
      aplicarClase(clase_local);

      Serial.print(F("Biceps:")); Serial.print(mvc_pct0, 1);
      Serial.print(F(",Triceps:")); Serial.print(mvc_pct1, 1);
      Serial.print(F(",Clase:")); Serial.println(clase_local);
    }

    // Mover servo con limitador de tasa (ambos modos)
    actualizarServo();

    // Log cada 500ms
    static uint32_t t_log = 0;
    if (millis() - t_log >= 500) {
      t_log = millis();
      const char* nombres[] = {"REPOSO", "FLEXION", "EXTENSION"};
      Serial.print(F("# ["));
      Serial.print(nombres[clase_actual]);
      Serial.print(F("] Bic="));  Serial.print(mvc_pct0, 1);
      Serial.print(F("% Tri="));  Serial.print(mvc_pct1, 1);
      Serial.print(F("% Ang="));  Serial.print(angulo_actual, 1);
      Serial.println(F("°"));
    }
  }
}
