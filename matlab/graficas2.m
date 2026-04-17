% --- CONFIGURACIÓN DE EXPORTACIÓN ---
output_dir = 'C:\Users\david\OneDrive\Desktop\Proyecto de titulo\Bicep\Figuras\';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

% --- CARGA Y PROCESAMIENTO ---
data_fuerza          = readtable('C:\Users\david\OneDrive\Desktop\Proyecto de titulo\Bicep\Fuerza\emg_5seg_20260129_225809.csv');
raw_fuerza           = data_fuerza.emg_raw;
fs                   = 1000;
t                    = (0:length(raw_fuerza)-1) / fs;

raw_fuerza_centered  = raw_fuerza - mean(raw_fuerza);
[bn, an]             = iirnotch(50/(fs/2), (50/(fs/2))/35);
f_clean              = filtfilt(bn, an, raw_fuerza_centered);
window_samples       = round(0.150 * fs);
rms_fuerza           = envelope(f_clean, window_samples, 'rms');
[valor_mvc, idx_mvc] = max(rms_fuerza);

fprintf('VALOR MVC CALCULADO: %.6f\n', valor_mvc);

% --- CONFIGURACIÓN VISUAL COMPARTIDA ---
ancho_cm  = 16;   % ancho de cada figura (ajustado a slide 16:9)
alto_cm   = 7;    % alto de cada figura
dpi       = 150;  % resolución para Beamer

estilos = struct();
estilos(1).senal   = raw_fuerza_centered;
estilos(1).color   = [0.55 0.55 0.55];
estilos(1).grosor  = 0.8;
estilos(1).nombre  = 'emg_cruda';
estilos(1).titulo  = 'Señal EMG Cruda (centrada)';
estilos(1).ylabel  = 'Amplitud (mV)';

estilos(2).senal   = f_clean;
estilos(2).color   = [0.0 0.50 0.0];
estilos(2).grosor  = 0.8;
estilos(2).nombre  = 'emg_filtrada';
estilos(2).titulo  = 'Señal Filtrada – Notch 50 Hz (Butterworth 4°)';
estilos(2).ylabel  = 'Amplitud (mV)';

estilos(3).senal   = rms_fuerza;
estilos(3).color   = [0.10 0.30 0.90];
estilos(3).grosor  = 2.5;
estilos(3).nombre  = 'emg_rms';
estilos(3).titulo  = 'Envolvente RMS (ventana 150 ms)';
estilos(3).ylabel  = 'Energía (mV)';

% =========================================================
%  FIGURAS 1–3: Cruda, Filtrada, RMS  (una por archivo)
% =========================================================
for k = 1:3
    fig = figure('Name', estilos(k).nombre, ...
        'Units', 'centimeters', 'Position', [2 2 ancho_cm alto_cm], ...
        'Color', 'w');

    plot(t, estilos(k).senal, ...
        'Color',     estilos(k).color, ...
        'LineWidth', estilos(k).grosor);

    title(estilos(k).titulo,  'FontSize', 11, 'FontWeight', 'bold');
    xlabel('Tiempo (s)',       'FontSize', 10);
    ylabel(estilos(k).ylabel,  'FontSize', 10);
    xlim([t(1) t(end)]);
    grid on;
    ax = gca;
    ax.GridAlpha    = 0.25;
    ax.FontSize     = 9;
    ax.Box          = 'on';
    ax.Color        = 'w';          % fondo del eje blanco

    exportgraphics(fig, fullfile(output_dir, [estilos(k).nombre '.png']), ...
        'Resolution', dpi, 'BackgroundColor', 'white');

    fprintf('✔ Exportada: %s.png\n', estilos(k).nombre);
    close(fig);
end

% =========================================================
%  FIGURA 4: Punto MVC sobre la envolvente RMS
% =========================================================
fig4 = figure('Name', 'emg_mvc', ...
    'Units', 'centimeters', 'Position', [2 2 ancho_cm alto_cm], ...
    'Color', 'w');

plot(t, rms_fuerza, 'Color', [0.10 0.30 0.90], 'LineWidth', 2.5); hold on;

% Línea vertical en el instante MVC
xline(t(idx_mvc), '--r', 'LineWidth', 1.2);

% Marcador del punto máximo
plot(t(idx_mvc), valor_mvc, 'ko', ...
    'MarkerFaceColor', [1.0 0.85 0.0], ...   % amarillo
    'MarkerSize', 10, 'LineWidth', 2);

% Etiqueta con el valor numérico
text(t(idx_mvc) + 0.05, valor_mvc, ...
    sprintf('  MVC = %.4f mV', valor_mvc), ...
    'FontSize', 9, 'Color', [0.7 0 0], 'FontWeight', 'bold');

title('Identificación del Punto MVC', 'FontSize', 11, 'FontWeight', 'bold');
xlabel('Tiempo (s)',   'FontSize', 10);
ylabel('Energía (mV)', 'FontSize', 10);
xlim([t(1) t(end)]);
grid on;
ax = gca;
ax.GridAlpha = 0.25;
ax.FontSize  = 9;
ax.Box       = 'on';
ax.Color     = 'w';
hold off;

exportgraphics(fig4, fullfile(output_dir, 'emg_mvc.png'), ...
    'Resolution', dpi, 'BackgroundColor', 'white');

fprintf('✔ Exportada: emg_mvc.png\n');
close(fig4);

fprintf('\n✅ 4 imágenes listas en:\n   %s\n', output_dir);