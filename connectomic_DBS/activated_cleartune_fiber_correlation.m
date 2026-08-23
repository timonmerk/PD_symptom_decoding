addpath(genpath("/Users/Timon/Documents/MATLAB/leaddbs"));


%% Visualizations
% By Patricia, October 16 2025

clear 
clc

% cd '/Volumes/daten/netstim/PROJECTS/patriciazvarova/DynamicCleartune/Timon_results_2share'

%define colors
colorBrCT = ([55, 110, 180] ./ 255); % ([136, 11, 164] ./ 255); % color for bradykinesia CT
colorBrT = colorBrCT * 0.6 + [1 1 1] * 0.5; % color for bradykinesia Timon

colorTrCT = ([41, 175, 127] ./ 255); % ([83, 191, 153] ./ 255); % color for tremor CT
colorTrT = colorTrCT * 0.6 + [1 1 1] * 0.5; % color for tremor Timon

colorAmpCT = ([223, 74, 74] ./ 255); % ([186, 200, 29] ./ 255); % color for stimu amplitude by CT
colorDyT = colorAmpCT * 0.6 + [1 1 1] * 0.5; % color for dyskinesia
colorDyT = [0.8745, 0.2902, 0.2902];

%% Left hemisphere
load('Organized_results/df_rcs02_left.mat');

%% === COMPOSITE PLOT: 2 rows × 3 columns ===
figure('Units','centimeters','Position',[5 5 27 20]);
tiledlayout(2,3,'TileSpacing','compact','Padding','compact');

%% === COLORS (already defined earlier)
% colorBrCT, colorBrT
% colorTrCT, colorTrT
% colorAmpCT, colorDyT

%% --- (1,1) Bradykinesia correlation ---
nexttile(1)
scatter(df.CT_B, df.Timon_B, 25, colorBrCT, 'filled', 'MarkerFaceAlpha',0.6); hold on
p = polyfit(df.CT_B, df.Timon_B, 1);
xfit = linspace(min(df.CT_B), max(df.CT_B), 100);
yfit = polyval(p, xfit);
plot(xfit, yfit, 'Color', colorBrCT, 'LineWidth', 2);
xlabel('Cleartune (% of fibers)')
ylabel('Timon')
title('Bradykinesia correlation')
box off; grid off;

%% --- (1,2) Tremor correlation ---
nexttile(2)
scatter(df.CT_T, df.Timon_T, 25, colorTrCT, 'filled', 'MarkerFaceAlpha',0.6); hold on
p = polyfit(df.CT_T, df.Timon_T, 1);
xfit = linspace(min(df.CT_T), max(df.CT_T), 100);
yfit = polyval(p, xfit);
plot(xfit, yfit, 'Color', colorTrCT, 'LineWidth', 2);
xlabel('Cleartune (% of fibers)')
ylabel('Timon')
title('Tremor correlation')
box off; grid off;

%% --- (1,3) Dyskinesia correlation ---
nexttile(3)
scatter(df.CT_Amplitude, df.Timon_D, 25, colorAmpCT, 'filled', 'MarkerFaceAlpha',0.6); hold on
p = polyfit(df.CT_Amplitude, df.Timon_D, 1);
xfit = linspace(min(df.CT_Amplitude), max(df.CT_Amplitude), 100);
yfit = polyval(p, xfit);
plot(xfit, yfit, 'Color', colorAmpCT, 'LineWidth', 2);
xlabel('CT Amplitude')
ylabel('Dyskinesia')
title('Dyskinesia correlation')
box off; grid off;

%% --- (2,1) Bradykinesia time series ---
nexttile(4)
plot(zscore(df.CT_B),'LineWidth',2.5,'Color',colorBrCT); hold on
plot(zscore(df.Timon_B),'LineWidth',3.5,'Color',colorBrT);
xlabel('Time (samples)')
ylabel('Z-scored value')
legend("Cleartune","Timon",'Location','best')
title('Bradykinesia time series')
box off; grid on

%% --- (2,2) Tremor time series ---
nexttile(5)
plot(zscore(df.CT_T),'LineWidth',2.5,'Color',colorTrCT); hold on
plot(zscore(df.Timon_T),'LineWidth',3.5,'Color',colorTrT);
xlabel('Time (samples)')
ylabel('Z-scored value')
legend("Cleartune","Timon",'Location','best')
title('Tremor time series')
box off; grid off;

%% --- (2,3) Dyskinesia time series ---
nexttile(6)
plot(zscore(df.CT_Amplitude),'LineWidth',2.5,'Color',colorAmpCT); hold on
plot(zscore(df.Timon_D),'LineWidth',3.5,'Color',colorDyT);
xlabel('Time (samples)')
ylabel('Z-scored value')
legend("Cleartune","Timon",'Location','best')
title('Dyskinesia time series')
box off; grid off;

%% === Export ===
exportgraphics(gcf,'Plots/rcs02_L/composite_corr_timeSeries_simple.pdf','Resolution',300);