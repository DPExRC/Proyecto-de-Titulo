// =============================================================================
// EMG v4.1 — PUENTE DE ADQUISICIÓN PURO (DAQ) — TOTALMENTE ASÍNCRONO
// =============================================================================

#include <avr/interrupt.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <HardwareSerial.h>

// ---------------------------------------------------------------------------
// PCA9685
// ---------------------------------------------------------------------------
Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(0x40);
#define CANAL_SERVO_CODO    0
#define CANAL_SERVO_MUNECA  1

#define PULSO_MIN_US     500
#define PULSO_MAX_US     2400
#define FRECUENCIA_PWM   50

#define US_POR_CUENTA    (20000.0f / 4096.0f)
#define PULSO_MIN_CNT    ((uint16_t)(PULSO_MIN_US / US_POR_CUENTA))
#define PULSO_MAX_CNT    ((uint16_t)(PULSO_MAX_US / US_POR_CUENTA))

// ---------------------------------------------------------------------------
// PINES EMG — 3 canales
// ---------------------------------------------------------------------------
#define PIN_BICEPS     A0
#define PIN_TRICEPS    A1
#define PIN_ANTEBRAZO  A2   

#define N_CANALES 3

// ---------------------------------------------------------------------------
// PARÁMETROS DE MUESTREO
// ---------------------------------------------------------------------------
#define FS_TOTAL_HZ      1000   
#define PASO_MS          20     

// ---------------------------------------------------------------------------
// LIMITADOR DE TASA (slew-rate)
// ---------------------------------------------------------------------------
#define VEL_MAX_SERVO    300.0f
#define DT_CONTROL       (PASO_MS / 1000.0f)
#define MAX_CAMBIO       (VEL_MAX_SERVO * DT_CONTROL * 0.8f)  // 4.8°/ciclo

// ---------------------------------------------------------------------------
// ESTADO DE MUESTREO — Volatile para comunicación con ISR
// ---------------------------------------------------------------------------
volatile int16_t muestra_biceps    = 0;
volatile int16_t muestra_triceps   = 0;
volatile int16_t muestra_antebrazo = 0;
volatile bool    trama_lista       = false;
volatile uint8_t canal_actual      = 0;

// ---------------------------------------------------------------------------
// ESTADO DE CONTROL — ángulos objetivo y actuales
// ---------------------------------------------------------------------------
float angulo_codo_actual    = 0.0f;
float angulo_codo_meta      = 0.0f;
float angulo_muneca_actual  = 0.0f;
float angulo_muneca_meta    = 0.0f;

volatile bool comando_procesado = false;

// ---------------------------------------------------------------------------
// DIAGNÓSTICO — duración real de la ISR (reemplaza el análisis estático de
// ciclos de instrucción: se mide con micros(), no se estima de forma
// teórica sobre el ensamblador compilado).
// ---------------------------------------------------------------------------
volatile uint16_t isr_duracion_ultima_us = 0;
volatile uint16_t isr_duracion_max_us    = 0;

// ---------------------------------------------------------------------------
// TIMER1 — Dispara a 1000 Hz total
// ---------------------------------------------------------------------------
void configurarTimer1() {
  cli();
  TCCR1A = 0; TCCR1B = 0; TCNT1 = 0;
  OCR1A  = 1999;             // 16MHz / (8 * 1000Hz) - 1 = 1999
  TCCR1B |= (1 << WGM12);    // Modo CTC
  TCCR1B |= (1 << CS11);     // Prescaler 8
  TIMSK1 |= (1 << OCIE1A);   // Habilitar interrupción
  sei();
}

ISR(TIMER1_COMPA_vect) {
  uint16_t t_inicio_us = micros();

  switch (canal_actual) {
    case 0:
      muestra_biceps = analogRead(PIN_BICEPS);
      canal_actual = 1;
      break;
    case 1:
      muestra_triceps = analogRead(PIN_TRICEPS);
      canal_actual = 2;
      break;
    default:
      muestra_antebrazo = analogRead(PIN_ANTEBRAZO);
      canal_actual = 0;
      trama_lista = true; // Fin de ciclo de escaneo de los 3 canales
      break;
  }

  isr_duracion_ultima_us = (uint16_t)(micros() - t_inicio_us);
  if (isr_duracion_ultima_us > isr_duracion_max_us) {
    isr_duracion_max_us = isr_duracion_ultima_us;
  }
}

// ---------------------------------------------------------------------------
// SERVOS
// ---------------------------------------------------------------------------
void moverServo(uint8_t canal, float angulo) {
  angulo = constrain(angulo, 0.0f, 180.0f);
  uint16_t cuentas = (uint16_t)map(
    (long)(angulo * 10), 0, 1800, PULSO_MIN_CNT, PULSO_MAX_CNT
  );
  pca.setPWM(canal, 0, cuentas);
}

float actualizarAngulo(uint8_t canal_pca, float actual, float meta) {
  float dif = meta - actual;
  if (abs(dif) > MAX_CAMBIO)
    actual += (dif > 0 ? 1.0f : -1.0f) * MAX_CAMBIO;
  else
    actual = meta;
  actual = constrain(actual, 0.0f, 180.0f);
  moverServo(canal_pca, actual);
  return actual;
}

// ---------------------------------------------------------------------------
// PARSER DE COMANDOS SERIALES (No bloqueante)
// ---------------------------------------------------------------------------
char rx_buf[24];
uint8_t rx_idx = 0;

void procesarLinea(char *linea) {
  if (linea[0] != 'A' || linea[1] != ',') return;
  char *resto = linea + 2;
  char *coma = strchr(resto, ',');
  if (coma == NULL) return;   

  *coma = '\0';
  float codo   = atof(resto);
  float muneca = atof(coma + 1);

  angulo_codo_meta   = constrain(codo, 0.0f, 180.0f);
  angulo_muneca_meta = constrain(muneca, 0.0f, 180.0f);
  
  comando_procesado = true;
}

void leerSerial() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      rx_buf[rx_idx] = '\0';
      rx_idx = 0;
      procesarLinea(rx_buf);
    } else if (rx_idx < sizeof(rx_buf) - 1) {
      rx_buf[rx_idx++] = c;
    }
  }
}

// ---------------------------------------------------------------------------
// TRANSMISIÓN PROTEGIDA (Copia atómica de 16-bits)
// ---------------------------------------------------------------------------
void transmitirTrama() {
  int16_t local_biceps, local_triceps, local_antebrazo;

  // Bloqueo de interrupción ultracorto para clonar las variables de forma segura
  uint8_t sreg_backup = SREG;
  cli();
  local_biceps    = muestra_biceps;
  local_triceps   = muestra_triceps;
  local_antebrazo = muestra_antebrazo;
  SREG = sreg_backup; 

  Serial.print(F("S,"));
  Serial.print(local_biceps);    Serial.print(F(","));
  Serial.print(local_triceps);   Serial.print(F(","));
  Serial.println(local_antebrazo);
}

void transmitirACK() {
  Serial.println(F("A"));
}

// ---------------------------------------------------------------------------
// SETUP
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  Wire.begin();
  Wire.setClock(400000); // I2C a 400kHz rápido
  
  pca.begin();
  pca.setOscillatorFrequency(25000000);
  pca.setPWMFreq(FRECUENCIA_PWM);
  delay(10);

  // Diagnóstico I2C real: Wire.endTransmission() devuelve 0 si el PCA9685
  // respondió en el bus. Se imprime como línea '#...' — esperar_ready() en
  // capture.py ya tolera y muestra líneas con ese prefijo antes de READY,
  // así que no cambia el protocolo de arranque existente.
  Wire.beginTransmission(0x40);
  uint8_t error_i2c = Wire.endTransmission();
  Serial.print(F("#I2C_PCA9685:"));
  Serial.println(error_i2c == 0 ? F("OK") : String(error_i2c));

  moverServo(CANAL_SERVO_CODO, 0.0f);
  moverServo(CANAL_SERVO_MUNECA, 0.0f);

  pinMode(PIN_BICEPS,    INPUT);
  pinMode(PIN_TRICEPS,   INPUT);
  pinMode(PIN_ANTEBRAZO, INPUT);

  configurarTimer1();

  Serial.println(F("READY"));
}

// ---------------------------------------------------------------------------
// LOOP PRINCIPAL (Estructura de ejecución libre)
// ---------------------------------------------------------------------------
void loop() {
  // 1. Leer el puerto constantemente sin importar si hay trama lista o no
  leerSerial();

  // 2. Transmitir datos de forma asíncrona solo cuando la ISR cambie el flag
  if (trama_lista) {
    trama_lista = false;
    transmitirTrama(); 
  }

  // 3. Ventana temporal exacta para el control de los servos y envío de ACK
  static uint32_t ultimo_control = 0;
  uint32_t ahora = millis();
  if (ahora - ultimo_control >= PASO_MS) {
    ultimo_control += PASO_MS; // Evita deriva temporal acumulada

    angulo_codo_actual = actualizarAngulo(
      CANAL_SERVO_CODO, angulo_codo_actual, angulo_codo_meta);
    angulo_muneca_actual = actualizarAngulo(
      CANAL_SERVO_MUNECA, angulo_muneca_actual, angulo_muneca_meta);

    if (comando_procesado) {
      transmitirACK();
      comando_procesado = false;
    }
  }

  // 4. Diagnóstico de duración de la ISR, 1 vez/s — reemplaza el análisis
  //    estático de ciclos de instrucción por una medición real con
  //    micros(). Línea '#...', ya ignorada de forma segura por
  //    esperar_ready()/leer_trama() en el lado Python.
  static uint32_t ultimo_diag = 0;
  if (ahora - ultimo_diag >= 1000) {
    ultimo_diag = ahora;
    noInterrupts();
    uint16_t max_us = isr_duracion_max_us;
    interrupts();
    Serial.print(F("#ISR_MAX_US:"));
    Serial.println(max_us);
  }
}