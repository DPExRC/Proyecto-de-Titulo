// =============================================================================
// EMG BRIDGE V2
// Arduino UNO + PCA9685
//
// Mejoras sobre v1:
//   - Timestamp en TX (micros)
//   - Confirmación ACK en RX
//   - Frecuencia de muestreo configurable desde Python (CFG)
//   - Soporte automático A0–A5 sin cambiar lógica
//
// Formato TX:
//   EMG:123456,512,498
//   └── timestamp_us, canal0, canal1, ...
//
// Formato RX:
//   ANG:90.00      → mover servo, responde ACK:90.00
//   CFG:500        → cambiar período de muestreo a 500 Hz
//
// =============================================================================

#include <Wire.h>
#include <Adafruit_PWMServoDriver.h>

// -----------------------------------------------------------------------------
// CONFIGURACIÓN EMG
// -----------------------------------------------------------------------------

const uint8_t CANALES_EMG[] = {
    A0,
    A1
};

const uint8_t N_CANALES =
    sizeof(CANALES_EMG) / sizeof(CANALES_EMG[0]);

// -----------------------------------------------------------------------------
// SERIAL
// -----------------------------------------------------------------------------

#define BAUDRATE 115200

// -----------------------------------------------------------------------------
// PCA9685
// -----------------------------------------------------------------------------

Adafruit_PWMServoDriver pca = Adafruit_PWMServoDriver(0x40);

#define SERVO_CANAL   0
#define SERVO_MIN_US  500
#define SERVO_MAX_US  2400
#define SERVO_FREQ    50

// -----------------------------------------------------------------------------
// TIEMPOS
// -----------------------------------------------------------------------------

// Período por defecto: 2000 µs = 500 Hz total = 250 Hz/canal (2 canales)
unsigned long periodoMuestreoUs = 2000UL;
unsigned long ultimoMuestreo    = 0;

// -----------------------------------------------------------------------------
// BUFFER RX
// -----------------------------------------------------------------------------

char    rxBuffer[32];
uint8_t rxIndex = 0;

// -----------------------------------------------------------------------------
// HELPERS
// -----------------------------------------------------------------------------

uint16_t microsegundosAPWM(uint16_t us)
{
    return (uint16_t)((float)us * SERVO_FREQ * 4096.0f / 1000000.0f);
}

void moverServo(float angulo)
{
    angulo = constrain(angulo, 0.0f, 180.0f);

    uint16_t us  = SERVO_MIN_US + (uint16_t)((angulo / 180.0f) * (SERVO_MAX_US - SERVO_MIN_US));
    uint16_t pwm = microsegundosAPWM(us);

    pca.setPWM(SERVO_CANAL, 0, pwm);
}

// -----------------------------------------------------------------------------
// TX — envío de muestra EMG con timestamp
// -----------------------------------------------------------------------------

void enviarEMG()
{
    // Timestamp primero, luego canales separados por coma
    Serial.print("EMG:");
    Serial.print(micros());

    for (uint8_t i = 0; i < N_CANALES; i++)
    {
        Serial.print(',');
        Serial.print(analogRead(CANALES_EMG[i]));
    }

    Serial.println();
}

// -----------------------------------------------------------------------------
// RX — parseo de comandos
// -----------------------------------------------------------------------------

void procesarComando(char* linea)
{
    // ANG:90.00 → mover servo + ACK
    if (strncmp(linea, "ANG:", 4) == 0)
    {
        float angulo = atof(linea + 4);
        moverServo(angulo);

        // Confirmación al host Python
        Serial.print("ACK:");
        Serial.println(angulo, 2);
        return;
    }

    // CFG:500 → cambiar frecuencia de muestreo (Hz)
    if (strncmp(linea, "CFG:", 4) == 0)
    {
        uint16_t hz = (uint16_t)atoi(linea + 4);
        if (hz >= 10 && hz <= 2000)
        {
            periodoMuestreoUs = 1000000UL / hz;
            Serial.print("CFG:OK:");
            Serial.println(hz);
        }
        else
        {
            Serial.println("CFG:ERR");
        }
        return;
    }
}

void leerSerial()
{
    while (Serial.available())
    {
        char c = Serial.read();

        if (c == '\r') continue;

        if (c == '\n')
        {
            rxBuffer[rxIndex] = '\0';
            procesarComando(rxBuffer);
            rxIndex = 0;
        }
        else
        {
            if (rxIndex < sizeof(rxBuffer) - 1)
                rxBuffer[rxIndex++] = c;
        }
    }
}

// -----------------------------------------------------------------------------
// SETUP / LOOP
// -----------------------------------------------------------------------------

void setup()
{
    Serial.begin(BAUDRATE);

    Wire.begin();
    Wire.setClock(400000);

    pca.begin();
    pca.setOscillatorFrequency(25000000);
    pca.setPWMFreq(SERVO_FREQ);

    delay(500);
    moverServo(0);

    Serial.println("READY");   // señal de arranque para Python
}

void loop()
{
    leerSerial();

    unsigned long ahora = micros();

    if (ahora - ultimoMuestreo >= periodoMuestreoUs)
    {
        ultimoMuestreo = ahora;
        enviarEMG();
    }
}
