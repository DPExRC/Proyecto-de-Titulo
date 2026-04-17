% --- VISUALIZACIÓN INTEGRADA CON CONTRASTE ALTO Y FONDO BLANCO ---
data_fuerza = readtable('C:\Users\david\OneDrive\Desktop\Proyecto de titulo\Bicep\Fuerza\emg_5seg_20260129_225809.csv');
raw_fuerza = data_fuerza.emg_raw;
fs = 1000;
t = (0:length(raw_fuerza)-1)/fs;

% 1. PROCESAMIENTO
raw_fuerza_centered = raw_fuerza - mean(raw_fuerza);
[bn, an] = iirnotch(50/(fs/2), (50/(fs/2))/35);
f_clean = filtfilt(bn, an, raw_fuerza_centered);
window_samples = round(0.150 * fs);
rms_fuerza = envelope(f_clean, window_samples, 'rms');
[valor_mvc, idx_mvc] = max(rms_fuerza);

% 2. GRAFICACIÓN
figure('Name', 'Análisis MVC - Contraste de Señales', 'Color', 'w');  % ← fondo figura blanco

% --- CAPA 1: SEÑAL CRUDA ---
plot(t, raw_fuerza_centered, 'Color', [0.7 0.7 0.7], 'LineWidth', 0.5); hold on;
% --- CAPA 2: SEÑAL FILTRADA ---
plot(t, f_clean, 'Color', [0.0 0.5 0.0], 'LineWidth', 0.5);
% --- CAPA 3: ENVOLTURA RMS ---
plot(t, rms_fuerza, 'b', 'LineWidth', 3.5);
% --- CAPA 4: PUNTO MVC ---
plot(t(idx_mvc), valor_mvc, 'ko', 'MarkerFaceColor', 'y', 'MarkerSize', 12, 'LineWidth', 2);

% 3. ESTÉTICA Y AJUSTES
title('Procesamiento de Señal EMG para Obtención de MVC', 'FontSize', 12);
xlabel('Tiempo (s)'); ylabel('Amplitud');
legend('1. Cruda (Centrada)', '2. Filtrada (Notch)', ...
    '3. RMS (ENERGÍA)', '4. PUNTO MVC', ...
    'Location', 'northeastoutside');
grid on;
ax = gca;
ax.Color      = 'w';      % ← fondo del área de ejes blanco
ax.GridAlpha  = 0.3;
ax.GridColor  = [0 0 0];  % ← cuadrícula negra (visible sobre blanco)
ax.XColor     = 'k';      % ← ejes y etiquetas en negro
ax.YColor     = 'k';

ylim([min(f_clean)*1.2, max(rms_fuerza)*1.4]);
hold off;

fprintf('VALOR MVC CALCULADO: %.6f\n', valor_mvc);