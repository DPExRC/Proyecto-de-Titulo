% --- CONFIGURACIÓN DE EXPORTACIÓN ---
output_dir = 'C:\Users\david\OneDrive\Desktop\Proyecto de titulo\Bicep\Figuras\';
if ~exist(output_dir, 'dir'), mkdir(output_dir); end

% --- CARGA Y PROCESAMIENTO ---
% Nota: Asegúrate de que la ruta del CSV sea correcta
data_fuerza      = readtable('C:\Users\david\OneDrive\Desktop\Proyecto de titulo\Bicep\Fuerza\emg_5seg_20260129_225809.csv');
raw_fuerza       = data_fuerza.emg_raw;
fs               = 1000;
t                = (0:length(raw_fuerza)-1) / fs;

% Procesamiento de señal
raw_fuerza_centered  = raw_fuerza - mean(raw_fuerza);
[bn, an]             = iirnotch(50/(fs/2), (50/(fs/2))/35);
f_clean              = filtfilt(bn, an, raw_fuerza_centered);
window_samples       = round(0.150 * fs);
rms_fuerza           = envelope(f_clean, window_samples, 'rms');
[valor_mvc, idx_mvc] = max(rms_fuerza);

fprintf('VALOR MVC CALCULADO: %.6f\n', valor_mvc);

% --- CONFIGURACIÓN VISUAL COMPARTIDA ---
ancho_cm  = 16;   % Ajustado para slides 16:9
alto_cm   = 7;    
dpi       = 150;  

estilos = struct();
estilos(1).senal   = raw_fuerza_centered;
estilos(1).color   = [0.4 0.4 0.4]; % Gris oscuro para la cruda
estilos(1).grosor  = 0.6;
estilos(1).nombre  = 'emg_cruda';
estilos(1).titulo  = 'Señal EMG Cruda';
estilos(1).ylabel  = 'Amplitud (mV)';

estilos(2).senal   = f_clean;
estilos(2).color   = [0.0 0.5 0.0]; % Verde para filtrada
estilos(2).grosor  = 0.6;
estilos(2).nombre  = 'emg_filtrada';
estilos(2).titulo  = 'Señal Filtrada – Notch 50 Hz (Butterworth 4°)';
estilos(2).ylabel  = 'Amplitud (mV)';

estilos(3).senal   = rms_fuerza;
estilos(3).color   = [0.1 0.3 0.9]; % Azul para RMS
estilos(3).grosor  = 2.0;
estilos(3).nombre  = 'emg_rms';
estilos(3).titulo  = 'Envolvente RMS (ventana 150 ms)';
estilos(3).ylabel  = 'Energía (mV)';

% =========================================================
%  FIGURAS 1–3: Cruda, Filtrada, RMS
% =========================================================
for k = 1:3
    fig = figure('Name', estilos(k).nombre, ...
        'Units', 'centimeters', 'Position', [2 2 ancho_cm alto_cm], ...
        'Color', 'w', 'Visible', 'off'); % Fondo de la ventana blanco

    plot(t, estilos(k).senal, ...
        'Color',     estilos(k).color, ...
        'LineWidth', estilos(k).grosor);

    % Configuración de etiquetas y títulos en NEGRO
    title(estilos(k).titulo, 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
    xlabel('Tiempo (s)',     'FontSize', 10, 'Color', 'k');
    ylabel(estilos(k).ylabel, 'FontSize', 10, 'Color', 'k');
    
    xlim([t(1) t(end)]);
    grid on;
    
    % Configuración del objeto AXES (Ejes)
    ax = gca;
    ax.XColor = 'k';        % Eje X en negro puro
    ax.YColor = 'k';        % Eje Y en negro puro
    ax.GridAlpha = 0.2;     % Cuadrícula suave
    ax.FontSize = 9;
    ax.Box = 'on';
    ax.Color = 'w';         % Fondo del gráfico blanco (CORRECCIÓN)

    % Exportación
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
    'Color', 'w', 'Visible', 'off'); % Fondo de la ventana blanco

plot(t, rms_fuerza, 'Color', [0.1 0.3 0.9], 'LineWidth', 2.0); hold on;

% Línea vertical y punto máximo
xline(t(idx_mvc), '--r', 'LineWidth', 1.2);
plot(t(idx_mvc), valor_mvc, 'ko', 'MarkerFaceColor', [1.0 0.85 0.0], ...
    'MarkerSize', 8, 'LineWidth', 1.5);

% Etiqueta del valor
text(t(idx_mvc) + 0.08, valor_mvc, ...
    sprintf('MVC = %.4f mV', valor_mvc), ...
    'FontSize', 9, 'Color', [0.6 0 0], 'FontWeight', 'bold');

% Títulos y ejes en NEGRO
title('Identificación del Punto MVC (Máxima Contracción)', 'FontSize', 11, 'FontWeight', 'bold', 'Color', 'k');
xlabel('Tiempo (s)',  'FontSize', 10, 'Color', 'k');
ylabel('Energía (mV)', 'FontSize', 10, 'Color', 'k');

xlim([t(1) t(end)]);
grid on;

ax = gca;
ax.XColor = 'k'; 
ax.YColor = 'k';
ax.GridAlpha = 0.2;
ax.FontSize = 9;
ax.Box = 'on';
ax.Color = 'w';         % Fondo del gráfico blanco (AQUÍ ESTABA EL PROBLEMA)
hold off;

% Exportación Final
exportgraphics(fig4, fullfile(output_dir, 'emg_mvc.png'), ...
    'Resolution', dpi, 'BackgroundColor', 'white');

fprintf('✔ Exportada: emg_mvc.png\n');
close(fig4);

fprintf('\n✅ Proceso completado. Las 4 imágenes tienen fondo blanco uniforme.\n');