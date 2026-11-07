%% plot_fig3_class_distribution.m
% Figure 3 — Per-class image counts across the four FER datasets.
% Vertical grouped bars on a log y-axis. Reproduces the data shown in
% fig3_v5.png with the orientation flipped to vertical.
%
% Run from MATLAB:  >> plot_fig3_class_distribution
%
% Outputs (next to this script):
%   fig3_matlab.png   (300 dpi raster)
%   fig3_matlab.pdf   (vector)

clear; clc; close all;

%% --- Data ---------------------------------------------------------------
emotions = {'Happy','Sad','Angry','Surprise','Neutral'};
datasets = {'ExpW','FER2013','RAF-DB','FERPlus'};

% rows = emotions, cols = datasets (same order as `datasets`)
counts = [ ...
    30753  7215  5057  591;   % Happy
     9915  4830  2460  311;   % Sad
     3234  4097   867  225;   % Angry
     6917  3171  1619  267;   % Surprise
    31569  4965  3171  694];  % Neutral

%% --- Style --------------------------------------------------------------
% Palette matched to fig3_v5 (ExpW=teal, FER2013=orange, RAF-DB=slate, FERPlus=pink)
colors = [ ...
    0.376 0.749 0.659;   % ExpW    #60BFA8
    0.949 0.584 0.416;   % FER2013 #F2956A
    0.604 0.655 0.780;   % RAF-DB  #9AA7C7
    0.902 0.561 0.718];  % FERPlus #E68FB7

fig = figure('Color','w','Units','pixels','Position',[100 100 1400 760]);
ax  = axes(fig);
hold(ax,'on');

%% --- Bars ---------------------------------------------------------------
b = bar(ax, counts, 'grouped', 'EdgeColor','none', 'BarWidth', 0.9);
for k = 1:numel(b)
    b(k).FaceColor = colors(k,:);
end

%% --- Axes ---------------------------------------------------------------
set(ax, 'YScale','log');
ylim(ax, [100 1e5]);
yticks(ax, [100 1000 10000 100000]);
yticklabels(ax, {'100','1,000','10,000','100,000'});
ax.YAxis.Exponent = 0;

xticks(ax, 1:numel(emotions));
xticklabels(ax, emotions);

ax.FontName = 'Helvetica';
ax.FontSize = 13;
ax.FontWeight = 'bold';
ax.LineWidth = 1.1;
ax.TickDir = 'out';
ax.Box = 'on';

ylabel(ax, 'Number of Images', 'FontSize', 14, 'FontWeight','bold');

% Light gridlines on the log axis (major + minor)
ax.YGrid = 'on';
ax.YMinorGrid = 'on';
ax.GridAlpha = 0.25;
ax.MinorGridAlpha = 0.12;
ax.MinorGridLineStyle = '-';

%% --- Value labels above each bar ---------------------------------------
nGroups = size(counts,1);
nBars   = size(counts,2);
groupW  = min(0.8, nBars/(nBars+1.5));

for g = 1:nGroups
    for k = 1:nBars
        % x-centre of the k-th bar in the g-th group (MATLAB grouped-bar formula)
        x = g - groupW/2 + (2*k-1) * groupW/(2*nBars);
        y = counts(g,k);
        txt = sprintf('%s', addThousandsSep(y));
        text(ax, x, y*1.08, txt, ...
            'HorizontalAlignment','center', ...
            'VerticalAlignment','bottom', ...
            'FontSize', 9, 'FontWeight','bold');
    end
end

%% --- Legend -------------------------------------------------------------
lg = legend(ax, datasets, 'Location','northwest', 'NumColumns', 2);
lg.FontSize = 12;
lg.Box = 'on';

%% --- Export -------------------------------------------------------------
% Use exportgraphics for crisp output (R2020a+); fall back to print otherwise.
outPng = fullfile(fileparts(mfilename('fullpath')), 'fig3_matlab.png');
outPdf = fullfile(fileparts(mfilename('fullpath')), 'fig3_matlab.pdf');
try
    exportgraphics(fig, outPng, 'Resolution', 300, 'BackgroundColor','white');
    exportgraphics(fig, outPdf, 'ContentType','vector', 'BackgroundColor','white');
catch
    print(fig, outPng, '-dpng', '-r300');
    print(fig, outPdf, '-dpdf', '-bestfit');
end
fprintf('Saved:\n  %s\n  %s\n', outPng, outPdf);

%% --- helpers ------------------------------------------------------------
function s = addThousandsSep(n)
    s = sprintf('%d', round(n));
    % insert commas every 3 digits from the right
    s = regexprep(s, '\B(?=(\d{3})+(?!\d))', ',');
end
