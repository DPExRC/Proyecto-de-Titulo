// =============================================================================
// EMG v4.0 — PUENTE DE ADQUISICIÓN PURO (DAQ) — 3 CANALES, 2 SERVOS
// =============================================================================
// Hardware:   Arduino Uno (ATmega328P, 16 MHz)
//             Módulo EMG genérico x3: bíceps → A0, tríceps → A1,
//                                      pronator teres (antebrazo) → A2
//             PCA9685 (I2C, 25 MHz): SDA=A4, SCL=A5
//             Servo codo   → canal 0 del PCA9685
//             Servo muñeca → canal 1 del PCA9685
//
// Arquitectura: el Arduino NO ejecuta DSP ni calibración. Su única
// responsabilidad es:
//   1. Muestrear los 3 canales ADC de forma sincronizada.
//   2. Transmitir las muestras crudas por serial.
//   3. Recibir 2 ángulos objetivo desde la PC y moverlos con un
//      limitador de tasa (slew-rate) como única protección mecánica
//      local.
// Todo el filtrado, ventaneo, extracción de características y la
// inferencia del regresor ocurren en Python (ver src/processing/ y
// src/inference/ en el repositorio).
//
// Fs:         1000 Hz total, alternado entre 3 canales (~333 Hz/canal)
// Protocolo TX (Arduino → PC):  S,<adc_biceps>,<adc_triceps>,<adc_antebrazo>\n
// Protocolo RX (PC → Arduino):  A,<angulo_codo>,<angulo_muneca>\n
// Serial:     115200 baud
// =============================================================================

#include <avr/interrupt.h>
#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>
#include <io90pwm1.h>
#include <HardwareSerial.h>
#include <USBAPI.h>

// ---------------------------------------------------------------------------
// PCA9685
// --------------------------------------------------------------------------
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
#define PIN_ANTEBRAZO  A2   // pronator teres

#define N_CANALES 3

// ---------------------------------------------------------------------------
// PARÁMETROS DE MUESTREO
// ---------------------------------------------------------------------------
#define FS_TOTAL_HZ      1000   // Hz, tasa total del ADC (los 3 canales
                                 // se leen en sucesión dentro de cada ciclo)
#define PASO_MS          20     // cadencia de recepción de comandos de ángulo

// ---------------------------------------------------------------------------
// LIMITADOR DE TASA (slew-rate) — única protección mecánica local
// ---------------------------------------------------------------------------
// KS-3518: 300°/s sin carga — margen 80%
#define VEL_MAX_SERVO    300.0f
#define DT_CONTROL       (PASO_MS / 1000.0f)
#define MAX_CAMBIO       (VEL_MAX_SERVO * DT_CONTROL * 0.8f)  // 4.8°/ciclo

// ---------------------------------------------------------------------------
// ESTADO DE MUESTREO — ISR alterna entre los 3 canales
// ---------------------------------------------------------------------------
volatile int16_t muestra_biceps    = 0;
volatile int16_t muestra_triceps   = 0;
volatile int16_t muestra_antebrazo = 0;
volatile bool    trama_lista       = false;  // true cuando los 3 canales
                                              // de un mismo ciclo están listos
volatile uint8_t canal_actual      = 0;      // 0=biceps,1=triceps,2=antebrazo

// ---------------------------------------------------------------------------
// ESTADO DE CONTROL — ángulos objetivo y actuales, 2 DOF
// ---------------------------------------------------------------------------
// Reposo = 0° en ambos DOF (convención fijada para todo el proyecto).
float angulo_codo_actual    = 0.0f;
float angulo_codo_meta      = 0.0f;
float angulo_muneca_actual  = 0.0f;
float angulo_muneca_meta    = 0.0f;

// ---------------------------------------------------------------------------
// TIMER1 — dispara a FS_TOTAL_HZ, cada disparo lee un canal y alterna
// Con 3 canales, cada canal individual se actualiza a FS_TOTAL_HZ/3 ≈ 333 Hz,
// consistente con el Nyquist efectivo usado en el pipeline Python.
// ---------------------------------------------------------------------------
void configurarTimer1() {
  cli();
  TCCR1A = 0; TCCR1B = 0; TCNT1 = 0;
  OCR1A  = 1999;                 // 1 ms → 1000 Hz total
  TCCR1B |= (1 << WGM12);
  TCCR1B |= (1 << CS11);         // prescaler 8
  TIMSK1 |= (1 << OCIE1A);
  sei();
}

ISR(TIMER1_COMPA_vect) {
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
      trama_lista = true;   // ciclo de 3 canales completo
      break;
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

// Aplica el limitador de tasa a un ángulo actual hacia su meta, y mueve
// el servo correspondiente. Se llama una vez por DOF por ciclo de control.
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
// PARSER DE COMANDOS SERIALES DESDE PC
// Protocolo: "A,<angulo_codo>,<angulo_muneca>\n"
// ---------------------------------------------------------------------------
char rx_buf[24];
uint8_t rx_idx = 0;

void procesarLinea(char *linea) {
  if (linea[0] != 'A' || linea[1] != ',') return;

  // linea = "A,<codo>,<muneca>" — separar por la coma tras "A,"
  char *resto = linea + 2;
  char *coma = strchr(resto, ',');
  if (coma == NULL) return;   // trama incompleta, se descarta

  *coma = '\0';
  float codo   = atof(resto);
  float muneca = atof(coma + 1);

  angulo_codo_meta   = constrain(codo, 0.0f, 180.0f);
  angulo_muneca_meta = constrain(muneca, 0.0f, 180.0f);
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
    // Si se excede el buffer sin '\n', la trama se descarta implícitamente
    // al sobreescribirse rx_idx en el próximo '\n' válido.
  }
}

// ---------------------------------------------------------------------------
// TRANSMISIÓN DE MUESTRAS CRUDAS
// Protocolo: "S,<adc_biceps>,<adc_triceps>,<adc_antebrazo>\n"
// ---------------------------------------------------------------------------
void transmitirTrama() {
  Serial.print(F("S,"));
  Serial.print(muestra_biceps);    Serial.print(F(","));
  Serial.print(muestra_triceps);   Serial.print(F(","));
  Serial.println(muestra_antebrazo);
}

// ---------------------------------------------------------------------------
// SETUP
// ---------------------------------------------------------------------------
void setup() {
  Serial.begin(115200);
  Wire.begin();

  Wire.setClock(400000); 
  
  pca.begin();
  pca.setOscillatorFrequency(25000000);
  pca.setPWMFreq(FRECUENCIA_PWM);
  delay(10);

  // Posición segura inicial: reposo = 0° en ambos DOF
  moverServo(CANAL_SERVO_CODO, 0.0f);
  moverServo(CANAL_SERVO_MUNECA, 0.0f);

  pinMode(PIN_BICEPS,    INPUT);
  pinMode(PIN_TRICEPS,   INPUT);
  pinMode(PIN_ANTEBRAZO, INPUT);

  configurarTimer1();

  Serial.println(F("READY"));
  Serial.println(F("# EMG v4.0 — Puente de adquisición puro, 3 canales"));
  Serial.println(F("# Canales: A0=Biceps A1=Triceps A2=Antebrazo(pronator teres)"));
  Serial.println(F("# Fs=1000Hz total (~333Hz/canal) | Sin DSP ni calibracion embebida"));
  Serial.println(F("# Protocolo TX: S,adc_biceps,adc_triceps,adc_antebrazo"));
  Serial.println(F("# Protocolo RX: A,angulo_codo,angulo_muneca"));
  Serial.println(F("# Reposo = 0 grados en ambos DOF"));
  Serial.println(F("#"));
}

// ---------------------------------------------------------------------------
// LOOP PRINCIPAL
// ---------------------------------------------------------------------------
void loop() {
  leerSerial();

  if (!trama_lista) return;
  trama_lista = false;

  // 1. Transmitir muestras crudas del ciclo actual
  transmitirTrama();

  // 2. Actualizar ambos servos con limitador de tasa, hacia el último
  //    ángulo objetivo recibido de la PC (angulo_*_meta se actualiza de
  //    forma asíncrona en procesarLinea() cada vez que llega un comando)
  static uint32_t ultimo_control = 0;
  uint32_t ahora = millis();
  if (ahora - ultimo_control >= PASO_MS) {
    ultimo_control = ahora;
    angulo_codo_actual = actualizarAngulo(
      CANAL_SERVO_CODO, angulo_codo_actual, angulo_codo_meta);
    angulo_muneca_actual = actualizarAngulo(
      CANAL_SERVO_MUNECA, angulo_muneca_actual, angulo_muneca_meta);
  }
}
