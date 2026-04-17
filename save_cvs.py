import serial
import time
from datetime import datetime

PUERTO_SERIAL = 'COM4'
BAUDRATE = 115200
NOMBRE_ARCHIVO = f'emg_5seg_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'

def guardar_datos_emg():
    ser = None
    try:
        print(f"Conectando a {PUERTO_SERIAL}...")
        ser = serial.Serial(PUERTO_SERIAL, BAUDRATE, timeout=1)
        print(f"✓ Conectado a {PUERTO_SERIAL}")
        
        # Esperar más tiempo para que Arduino se inicialice
        print("Esperando inicialización del Arduino...")
        time.sleep(3)
        
        # Limpiar buffer
        ser.reset_input_buffer()
        
        print(f"✓ Guardando en: {NOMBRE_ARCHIVO}")
        print("\nCapturando 5 segundos...")
        print("(No presiones Ctrl+C hasta que termine)\n")
        print("-" * 60)
        
        with open(NOMBRE_ARCHIVO, 'w', encoding='utf-8') as archivo:
            contador = 0
            inicio = time.time()
            sin_datos_contador = 0
            
            while True:
                if ser.in_waiting > 0:
                    try:
                        linea = ser.readline().decode('utf-8', errors='ignore').strip()
                        
                        if linea:
                            sin_datos_contador = 0  # Reset contador
                            
                            # Verificar si es el fin de captura
                            if linea == "FIN_CAPTURA":
                                print("\n" + "-" * 60)
                                print("✓ Captura completada (5 segundos)")
                                break
                            
                            # Mostrar cada 100 muestras para no saturar pantalla
                            if contador % 100 == 0:
                                print(linea)
                            
                            archivo.write(linea + '\n')
                            archivo.flush()
                            contador += 1
                    except Exception as e:
                        print(f"Error leyendo línea: {e}")
                else:
                    sin_datos_contador += 1
                    
                    # Si no hay datos por mucho tiempo, advertir
                    if sin_datos_contador > 1000:
                        print("⚠ No se reciben datos del Arduino...")
                        print("  Verifica que el código esté subido correctamente")
                        break
                    
                    time.sleep(0.001)  # Pequeña pausa
                
                # Timeout de seguridad (10 segundos)
                if time.time() - inicio > 10:
                    print("\n⚠ Timeout: Captura excedió 10 segundos")
                    break
        
        print(f"✓ Total muestras capturadas: {contador}")
        print(f"✓ Archivo guardado: {NOMBRE_ARCHIVO}")
        
        if contador == 0:
            print("\n⚠ ADVERTENCIA: No se capturaron datos")
            print("Posibles causas:")
            print("  1. Arduino no está ejecutando el código")
            print("  2. Presiona el botón RESET en el Arduino")
            print("  3. Vuelve a subir el código al Arduino")
        
    except KeyboardInterrupt:
        print(f"\n⚠ Detenido manualmente")
        if 'contador' in locals():
            print(f"   Muestras capturadas: {contador}")
        
    except serial.SerialException as e:
        print(f"\n✗ ERROR Serial: {e}")
        print("  - Cierra Arduino IDE")
        print("  - Reconecta el Arduino")
        
    except Exception as e:
        print(f"✗ Error inesperado: {e}")
        
    finally:
        if ser and ser.is_open:
            ser.close()
            print("\n✓ Puerto serial cerrado")

if __name__ == "__main__":
    print("=" * 60)
    print(" CAPTURA EMG - 5 SEGUNDOS")
    print("=" * 60)
    print()
    guardar_datos_emg()