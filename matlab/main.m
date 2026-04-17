%% ========================================================================
%  PROYECTO DE TÍTULO: ANÁLISIS AUTOMATIZADO DE EMG (MVC & BASELINE)
%  ========================================================================
clear; clc; close all;

% 1. RUTAS DE ARCHIVOS DE REPOSO (Baseline)
rutas_reposo = { ...
    'C:\Users\david\OneDrive\Desktop\Proyecto de titulo\Bicep\Relajado\emg_5seg_20260129_224642.csv'
};

% 2. RUTAS DE ARCHIVOS DE FUERZA (MVC)
rutas_fuerza = { ...
    'C:\Users\david\OneDrive\Desktop\Proyecto de titulo\Bicep\Fuerza\emg_5seg_20260129_225809.csv'
};

%% ========================================================================
%  BLOQUE DE PROCESAMIENTO
%  ========================================================================
fs = 1000; 
window_samples = round(0.150 * fs); 
[bn, an] = iirnotch(50/(fs/2), (50/(fs/2))/35); 

% --- A. Cálculo de Ruido Base (Baseline) ---
lista_ruido = [];
for r = 1:length(rutas_reposo)
    data_r = readtable(rutas_reposo{r});
    raw_r = data_r.emg_raw - mean(data_r.emg_raw);
    clean_r = filtfilt(bn, an, raw_r);
    rms_r = envelope(clean_r, window_samples, 'rms');
    lista_ruido = [lista_ruido; mean(rms_r)];
end
ruido_base_global = mean(lista_ruido);

% --- B. Procesamiento de MVC con Gráfico Integrado ---
resumen_final = table();

for f = 1:length(rutas_fuerza)
    % Procesar Fuerza
    data_f = readtable(rutas_fuerza{f});
    raw_f = data_f.emg_raw - mean(data_f.emg_raw);
    clean_f = filtfilt(bn, an, raw_f);
    t = (0:length(raw_f)-1)/fs;
    
    % RMS de la fuerza sin limpiar todavía (para mostrarla en el gráfico)
    rms_f_bruto = envelope(clean_f, window_samples, 'rms');
    
    % RMS NETO (Restando el ruido)
    rms_neto = max(0, rms_f_bruto - ruido_base_global);
    [mvc_val, idx] = max(rms_neto);
    
    % Guardar Resultados
    [~, nombre] = fileparts(rutas_fuerza{f});
    nueva_fila = table({nombre}, mvc_val, ruido_base_global, ...
        'VariableNames', {'Archivo', 'MVC_Neto', 'Ruido_Base'});
    resumen_final = [resumen_final; nueva_fila];
    
    % --- GRAFICACIÓN CON BASELINE INTEGRADA ---
    figure('Name', ['Análisis MVC Completo: ' nombre]);
    
    % 1. Señal Cruda y Filtrada (Fondo)
    plot(t, raw_f, 'Color', [0.8 0.8 0.8], 'LineWidth', 0.5); hold on;
    plot(t, clean_f, 'Color', [0.4 0.7 0.4], 'LineWidth', 0.5);
    
    % 2. EL BASELINE (Línea roja que indica el nivel de ruido restado)
    yline(ruido_base_global, '--r', 'LineWidth', 2, 'Label', 'Nivel de Reposo');
    
    % 3. RMS NETO (La fuerza real)
    plot(t, rms_neto, 'b', 'LineWidth', 2.5);
    
    % 4. PUNTO MVC
    plot(t(idx), mvc_val, 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 12);
    
    title(['Sujeto: ', nombre, ' | MVC Neto: ', num2str(mvc_val)]);
    xlabel('Tiempo (s)'); ylabel('Amplitud');
    legend('Señal Cruda', 'Filtrada (Notch)', 'Umbral de Reposo (Resta)', 'RMS Neto (Fuerza)', 'Punto MVC');
    grid on; hold off;
end

% --- C. Mostrar Tabla ---
disp('Resultados de Procesamiento:');
disp(resumen_final);