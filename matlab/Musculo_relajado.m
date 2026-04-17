% =========================================================
%   PROCESAMIENTO BASELINE - ESTILO VISUAL MEJORADO
% =========================================================

% --- PARÁMETROS GLOBALES ---
fs = 1000;
fc_low  = 20;
fc_high = 450;
orden   = 4;
f_notch = 50;
window_rms = 0.150;

% --- CARGAR DATOS ---
data_repo = readtable('C:\Users\david\OneDrive\Desktop\Proyecto de titulo\Bicep\Relajado\emg_5seg_20260129_224642.csv');
raw_repo  = data_repo.emg_raw;
t_repo    = (0:length(raw_repo)-1) / fs;

% =========================================================
%   PIPELINE DE FILTRADO
% =========================================================
repo_centered = raw_repo - mean(raw_repo);
[bn, an]   = iirnotch(f_notch/(fs/2), (f_notch/(fs/2))/35);
repo_notch = filtfilt(bn, an, repo_centered);
[b, a]     = butter(orden, [fc_low fc_high]/(fs/2), 'bandpass');
repo_clean = filtfilt(b, a, repo_notch);
window_samples = round(window_rms * fs);
rms_repo       = envelope(repo_clean, window_samples, 'rms');
valor_reposo_final = mean(rms_repo);

% =========================================================
%   GRÁFICO - RÉPLICA CON EJES NEGROS Y LÍNEA HORIZONTAL
% =========================================================

% Punto para el marcador circular (al centro del tiempo)
idx_marcador = round(length(t_repo) / 2);
t_marcador = t_repo(idx_marcador);

fig = figure('Name', 'Identificación del Punto Baseline', ...
             'Color', [1 1 1], ...         
             'Position', [100, 100, 900, 450]); 

% Configuración de ejes en NEGRO
ax = axes('Color', 'none', ...
          'XColor', [0 0 0], ...      % Ejes color negro
          'YColor', [0 0 0], ...      % Ejes color negro
          'LineWidth', 1.2, ...
          'FontSize', 14, ...
          'FontWeight', 'bold');
hold on;

% 1. Línea de la señal (Azul vibrante y gruesa)
plot(t_repo, rms_repo, 'Color', [0 0.3 0.9], 'LineWidth', 3.5);

% 2. LÍNEA HORIZONTAL constante a la altura del Baseline (Roja punteada)
line([0 5], [valor_reposo_final valor_reposo_final], ...
    'Color', 'r', 'LineStyle', '--', 'LineWidth', 2);

% 3. Línea vertical punteada roja hacia el marcador
line([t_marcador t_marcador], [0 valor_reposo_final], ...
    'Color', 'r', 'LineStyle', ':', 'LineWidth', 1.5);

% 4. Marcador circular (Amarillo con borde negro)
plot(t_marcador, valor_reposo_final, 'o', ...
    'MarkerSize', 12, ...
    'MarkerEdgeColor', 'k', ...
    'MarkerFaceColor', [1 0.8 0], ...
    'LineWidth', 2.5);

% 5. Texto del valor (Rojo oscuro)
text(t_marcador + 0.1, valor_reposo_final + 2, ...
    sprintf('Baseline = %.4f mV', valor_reposo_final), ...
    'Color', [0.6 0 0], ...
    'FontSize', 14, ...
    'FontWeight', 'bold', ...
    'VerticalAlignment', 'bottom');

% 6. Etiquetas y Título en NEGRO
title('Identificación del Punto Baseline', 'Color', [0 0 0], ...
      'FontSize', 16, 'FontWeight', 'bold');
xlabel('Tiempo (s)', 'Color', [0 0 0]);
ylabel('Energía (mV)', 'Color', [0 0 0]);

% Ajustes de límites
xlim([0 5]);
ylim([0 70]); 

grid off; 
box off;