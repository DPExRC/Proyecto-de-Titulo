import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def calculate_rms(data):
    """
    Calcula el RMS (Root Mean Square) de una señal
    """
    return np.sqrt(np.mean(data**2))

def calculate_windowed_rms(data, window_size_ms=250, sampling_rate=1000):
    """
    Calcula el RMS en ventanas deslizantes
    
    Parameters:
    - data: array de datos EMG
    - window_size_ms: tamaño de ventana en milisegundos
    - sampling_rate: frecuencia de muestreo en Hz (default 1000 Hz)
    """
    window_size = int(window_size_ms * sampling_rate / 1000)
    step_size = window_size // 2  # 50% overlap
    
    rms_values = []
    timestamps = []
    
    for i in range(0, len(data) - window_size, step_size):
        window = data[i:i + window_size]
        rms_values.append(calculate_rms(window))
        timestamps.append(i + window_size // 2)
    
    return np.array(timestamps), np.array(rms_values)

def analyze_emg_file(filepath, label):
    """
    Analiza un archivo CSV de EMG
    """
    # Leer el CSV
    df = pd.read_csv(filepath)
    
    # Filtrar solo datos con electrodo conectado (lead_off = 0)
    df_valid = df[df['lead_off'] == 0]
    
    print(f"\n{'='*50}")
    print(f"Análisis: {label}")
    print(f"{'='*50}")
    print(f"Total de muestras: {len(df)}")
    print(f"Muestras válidas (lead_off=0): {len(df_valid)}")
    print(f"Muestras descartadas: {len(df) - len(df_valid)}")
    
    # Calcular RMS global
    emg_data = df_valid['emg_raw'].values
    rms_global = calculate_rms(emg_data)
    
    print(f"\nRMS Global: {rms_global:.2f}")
    print(f"Valor medio: {np.mean(emg_data):.2f}")
    print(f"Desviación estándar: {np.std(emg_data):.2f}")
    print(f"Valor mínimo: {np.min(emg_data):.2f}")
    print(f"Valor máximo: {np.max(emg_data):.2f}")
    
    # Calcular RMS en ventanas
    timestamps, rms_windowed = calculate_windowed_rms(emg_data, window_size_ms=250)
    print(f"\nRMS promedio (ventanas 250ms): {np.mean(rms_windowed):.2f}")
    print(f"RMS máximo (ventanas): {np.max(rms_windowed):.2f}")
    
    return {
        'data': emg_data,
        'rms_global': rms_global,
        'timestamps_windowed': timestamps,
        'rms_windowed': rms_windowed,
        'df': df_valid
    }

def plot_comparison(reposo_data, mvc_data):
    """
    Genera gráficos comparativos
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Señal en reposo
    axes[0, 0].plot(reposo_data['data'][:1000], linewidth=0.5)
    axes[0, 0].set_title('Señal EMG - Reposo (primeros 1000 puntos)')
    axes[0, 0].set_xlabel('Muestra')
    axes[0, 0].set_ylabel('Amplitud (raw)')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Señal MVC
    axes[0, 1].plot(mvc_data['data'][:1000], linewidth=0.5, color='red')
    axes[0, 1].set_title('Señal EMG - MVC (primeros 1000 puntos)')
    axes[0, 1].set_xlabel('Muestra')
    axes[0, 1].set_ylabel('Amplitud (raw)')
    axes[0, 1].grid(True, alpha=0.3)
    
    # RMS en ventanas - Reposo
    axes[1, 0].plot(reposo_data['timestamps_windowed'], 
                    reposo_data['rms_windowed'], linewidth=1.5)
    axes[1, 0].axhline(y=reposo_data['rms_global'], 
                       color='green', linestyle='--', 
                       label=f'RMS Global: {reposo_data["rms_global"]:.2f}')
    axes[1, 0].set_title('RMS en ventanas - Reposo')
    axes[1, 0].set_xlabel('Muestra')
    axes[1, 0].set_ylabel('RMS')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # RMS en ventanas - MVC
    axes[1, 1].plot(mvc_data['timestamps_windowed'], 
                    mvc_data['rms_windowed'], linewidth=1.5, color='red')
    axes[1, 1].axhline(y=mvc_data['rms_global'], 
                       color='green', linestyle='--', 
                       label=f'RMS Global: {mvc_data["rms_global"]:.2f}')
    axes[1, 1].set_title('RMS en ventanas - MVC')
    axes[1, 1].set_xlabel('Muestra')
    axes[1, 1].set_ylabel('RMS')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('emg_comparison.png', dpi=300, bbox_inches='tight')
    print("\n✓ Gráfico guardado como 'emg_comparison.png'")
    plt.show()

def calculate_mvc_percentage(reposo_rms, mvc_rms):
    """
    Calcula el porcentaje de MVC
    """
    percentage = (reposo_rms / mvc_rms) * 100
    return percentage

# ==============================================================================
# SCRIPT PRINCIPAL
# ==============================================================================

if __name__ == "__main__":
    print("="*50)
    print("ANÁLISIS DE SEÑALES EMG - CÁLCULO DE RMS")
    print("="*50)
    
    # Rutas de los archivos 
    archivo_reposo = 'Bicep/Relajado/emg_5seg_20260129_224642.csv'  
    archivo_mvc = 'Bicep/Fuerza/emg_5seg_20260129_225809.csv'        
    
    try:
        # Analizar reposo
        reposo = analyze_emg_file(archivo_reposo, "REPOSO / SIN FUERZA")
        
        # Analizar MVC
        mvc = analyze_emg_file(archivo_mvc, "MVC (Máxima Contracción Voluntaria)")
        
        # Comparación
        print(f"\n{'='*50}")
        print("COMPARACIÓN")
        print(f"{'='*50}")
        print(f"RMS Reposo: {reposo['rms_global']:.2f}")
        print(f"RMS MVC: {mvc['rms_global']:.2f}")
        print(f"Ratio MVC/Reposo: {mvc['rms_global']/reposo['rms_global']:.2f}x")
        print(f"Activación en reposo: {calculate_mvc_percentage(reposo['rms_global'], mvc['rms_global']):.2f}% del MVC")
        
        # Generar gráficos
        plot_comparison(reposo, mvc)
        
        # Guardar resultados en CSV
        results_df = pd.DataFrame({
            'Condición': ['Reposo', 'MVC'],
            'RMS_Global': [reposo['rms_global'], mvc['rms_global']],
            'Media': [np.mean(reposo['data']), np.mean(mvc['data'])],
            'Desv_Std': [np.std(reposo['data']), np.std(mvc['data'])],
            'Min': [np.min(reposo['data']), np.min(mvc['data'])],
            'Max': [np.max(reposo['data']), np.max(mvc['data'])]
        })
        
        results_df.to_csv('resultados_rms.csv', index=False)
        print("\n✓ Resultados guardados en 'resultados_rms.csv'")
        
    except FileNotFoundError as e:
        print(f"\n Error: No se encontró el archivo")
        print(f"   Asegúrate de que los archivos CSV estén en la misma carpeta que este script")
        print(f"   Rutas buscadas:")
        print(f"   - {archivo_reposo}")
        print(f"   - {archivo_mvc}")
    except Exception as e:
        print(f"\n Error durante el análisis: {e}")