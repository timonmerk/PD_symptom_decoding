addpath(genpath("/Users/Timon/Documents/MATLAB/leaddbs"));


%% Visualizations
% By Patricia, October 16 2025

clear 
clc

% cd '/Volumes/daten/netstim/PROJECTS/patriciazvarova/DynamicCleartune/Timon_results_2share'

%define colors
colorBrCT = ([136, 11, 164] ./ 255); % color for bradykinesia CT
colorBrT = colorBrCT * 0.6 + [1 1 1] * 0.5; % color for bradykinesia Timon

colorTrCT = ([83, 191, 153] ./ 255); % color for tremor CT
colorTrT = colorTrCT * 0.6 + [1 1 1] * 0.5; % color for tremor Timon

colorAmpCT = ([186, 200, 29] ./ 255); % color for stimu amplitude by CT
colorDyT = colorAmpCT * 0.6 + [1 1 1] * 0.5; % color for dyskinesia

%% Left hemisphere
load('Organized_results/df_rcs02_left.mat');

% corrplot for tremor
h=ea_corrplot(df.CT_T, df.Timon_T, 0, {'', 'Tremor Cleartune (% of fibers)', 'Tremor Timon'}, [], [], colorTrCT);
h.Position=[100 100 500 700];
exportgraphics(h, 'Plots/rcs02_L/Tremor_corrplot.pdf', 'Resolution', 300)

% time series for tremor
figure;
set(gcf, 'Units', 'centimeters', 'Position', [5 5 55 15])
plot(zscore(df.CT_T),'LineWidth', 2.5, 'Color',colorTrCT)
hold on;
plot(zscore(df.Timon_T),'LineWidth', 3.5, 'Color',colorTrT)

xlabel('Time (samples)')
ylabel('Value')
legend("Tremor Cleartune", "Tremor Timon")

exportgraphics(gcf, 'Plots/rcs02_L/Tremor_timeSeries.pdf', 'Resolution', 300)


% corrplot for bradykinesia
h1=ea_corrplot(df.CT_B, df.Timon_B, 0, {'', 'Bradykinesia Cleartune (% of fibers)', 'Bradykinesia Timon'}, [], [], colorBrCT);
h1.Position=[100 100 500 700];
exportgraphics(h1, 'Plots/rcs02_L/Bradykinesia_corrplot.pdf', 'Resolution', 300)

% time series for bradykinesia
figure;
set(gcf, 'Units', 'centimeters', 'Position', [5 5 55 15])
plot(zscore(df.CT_B),'LineWidth', 2.5, 'Color',colorBrCT)
hold on;
plot(zscore(df.Timon_B),'LineWidth', 3.5, 'Color',colorBrT)

xlabel('Time (samples)')
ylabel('Value')
legend("Bradykinesia Cleartune", "Bradykinesia Timon")

exportgraphics(gcf, 'Plots/rcs02_L/Bradykinesia_timeSeries.pdf', 'Resolution', 300)

%Dyskinesia corrplot
h2=ea_corrplot(df.CT_Amplitude, df.Timon_D, 0, {'', 'CT Amplitude', 'Dyskinesia Timon'}, [], [], colorAmpCT);
h2.Position=[100 100 500 700];
exportgraphics(h2, 'Plots/rcs02_L/Amplitude_dyskinesia_corrplot.pdf', 'Resolution', 300)

% time series for dyskinesia vs amplitude
figure;
set(gcf, 'Units', 'centimeters', 'Position', [5 5 55 15])
plot(zscore(df.CT_Amplitude),'LineWidth', 2.5, 'Color',colorAmpCT)
hold on;
plot(zscore(df.Timon_D),'LineWidth', 3.5, 'Color',colorDyT)

xlabel('Time (samples)')
ylabel('Value')
legend("Amplitude Cleartune", "Dyskinesia Timon")

exportgraphics(gcf, 'Plots/rcs02_L/Amplitude_dyskinesia_timeSeries.pdf', 'Resolution', 300)


%% right hemisphere
load('Organized_results/df_rcs02_right.mat');

% corrplot for tremor
h=ea_corrplot(df.CT_T, df.Timon_T, 0, {'', 'Tremor Cleartune (% of fibers)', 'Tremor Timon'}, [], [], colorTrCT);
h.Position=[100 100 500 700];
exportgraphics(h, 'Plots/rcs02_R/Tremor_corrplot.pdf', 'Resolution', 300)

% time series for tremor
figure;
set(gcf, 'Units', 'centimeters', 'Position', [5 5 55 15])
plot(zscore(df.CT_T),'LineWidth', 2.5, 'Color',colorTrCT)
hold on;
plot(zscore(df.Timon_T),'LineWidth', 3.5, 'Color',colorTrT)

xlabel('Time (samples)')
ylabel('Value')
legend("Tremor Cleartune", "Tremor Timon")

exportgraphics(gcf, 'Plots/rcs02_R/Tremor_timeSeries.pdf', 'Resolution', 300)


% corrplot for bradykinesia
h1=ea_corrplot(df.CT_B, df.Timon_B, 0, {'', 'Bradykinesia Cleartune (% of fibers)', 'Bradykinesia Timon'}, [], [], colorBrCT);
h1.Position=[100 100 500 700];
exportgraphics(h1, 'Plots/rcs02_R/Bradykinesia_corrplot.pdf', 'Resolution', 300)

% time series for bradykinesia
figure;
set(gcf, 'Units', 'centimeters', 'Position', [5 5 55 15])
plot(zscore(df.CT_B),'LineWidth', 2.5, 'Color',colorBrCT)
hold on;
plot(zscore(df.Timon_B),'LineWidth', 3.5, 'Color',colorBrT)

xlabel('Time (samples)')
ylabel('Value')
legend("Bradykinesia Cleartune", "Bradykinesia Timon")

exportgraphics(gcf, 'Plots/rcs02_R/Bradykinesia_timeSeries.pdf', 'Resolution', 300)

%Dyskinesia corrplot
h2=ea_corrplot(df.CT_Amplitude, df.Timon_D, 0, {'', 'CT Amplitude', 'Dyskinesia Timon'}, [], [], colorAmpCT);
h2.Position=[100 100 500 700];
exportgraphics(h2, 'Plots/rcs02_L/Amplitude_dyskinesia_corrplot.pdf', 'Resolution', 300)

% time series for dyskinesia vs amplitude
figure;
set(gcf, 'Units', 'centimeters', 'Position', [5 5 55 15])
plot(zscore(df.CT_Amplitude),'LineWidth', 2.5, 'Color',colorAmpCT)
hold on;
plot(zscore(df.Timon_D),'LineWidth', 3.5, 'Color',colorDyT)

xlabel('Time (samples)')
ylabel('Value')
legend("Amplitude Cleartune", "Dyskinesia Timon")

exportgraphics(gcf, 'Plots/rcs02_R/Amplitude_dyskinesia_timeSeries.pdf', 'Resolution', 300)

%% rcs02 Left Vercise directed
load('Organized_results/df_rcs02VerciseDirected_left.mat')


% corrplot for tremor
h=ea_corrplot(df.CT_T, df.Timon_T, 0, {'', 'Tremor Cleartune (% of fibers)', 'Tremor Timon'}, [], [], colorTrCT);
h.Position=[100 100 500 700];
exportgraphics(h, 'Plots/rcs02_L_VerciseDirected/Tremor_corrplot.png', 'Resolution', 300)

% time series for tremor
figure;
set(gcf, 'Units', 'centimeters', 'Position', [5 5 55 15])
plot(zscore(df.CT_T),'LineWidth', 2.5, 'Color',colorTrCT)
hold on;
plot(zscore(df.Timon_T),'LineWidth', 3.5, 'Color',colorTrT)

xlabel('Time (samples)')
ylabel('Value')
legend("Tremor Cleartune", "Tremor Timon")

exportgraphics(gcf, 'Plots/rcs02_L_VerciseDirected/Tremor_timeSeries.png', 'Resolution', 300)

% corrplot for tremor thr 700
h=ea_corrplot(df.CT_T_Thr700, df.Timon_T, 0, {'', 'Tremor Cleartune (% of fibers)', 'Tremor Timon'}, [], [], colorTrCT);
h.Position=[100 100 500 700];
exportgraphics(h, 'Plots/rcs02_L_VerciseDirected/Tremor_Thr700_corrplot.png', 'Resolution', 300)

% time series for tremor thr 700
figure;
set(gcf, 'Units', 'centimeters', 'Position', [5 5 55 15])
plot(zscore(df.CT_T_Thr700),'LineWidth', 2.5, 'Color',colorTrCT)
hold on;
plot(zscore(df.Timon_T),'LineWidth', 3.5, 'Color',colorTrT)

xlabel('Time (samples)')
ylabel('Value')
legend("Tremor Cleartune", "Tremor Timon")

exportgraphics(gcf, 'Plots/rcs02_L_VerciseDirected/Tremor_Thr700_timeSeries.png', 'Resolution', 300)


% corrplot for bradykinesia
h1=ea_corrplot(df.CT_B, df.Timon_B, 0, {'', 'Bradykinesia Cleartune (% of fibers)', 'Bradykinesia Timon'}, [], [], colorBrCT);
h1.Position=[100 100 500 700];
exportgraphics(h1, 'Plots/rcs02_L_VerciseDirected/Bradykinesia_corrplot.png', 'Resolution', 300)

% time series for bradykinesia
figure;
set(gcf, 'Units', 'centimeters', 'Position', [5 5 55 15])
plot(zscore(df.CT_B),'LineWidth', 2.5, 'Color',colorBrCT)
hold on;
plot(zscore(df.Timon_B),'LineWidth', 3.5, 'Color',colorBrT)

xlabel('Time (samples)')
ylabel('Value')
legend("Bradykinesia Cleartune", "Bradykinesia Timon")

exportgraphics(gcf, 'Plots/rcs02_L_VerciseDirected/Bradykinesia_timeSeries.png', 'Resolution', 300)

% corrplot for bradykinesia
h1=ea_corrplot(df.CT_B_Thr700, df.Timon_B, 0, {'', 'Bradykinesia Cleartune (% of fibers)', 'Bradykinesia Timon'}, [], [], colorBrCT);
h1.Position=[100 100 500 700];
exportgraphics(h1, 'Plots/rcs02_L_VerciseDirected/Bradykinesia_Thr700_corrplot.png', 'Resolution', 300)

% time series for bradykinesia
figure;
set(gcf, 'Units', 'centimeters', 'Position', [5 5 55 15])
plot(zscore(df.CT_B_Thr700),'LineWidth', 2.5, 'Color',colorBrCT)
hold on;
plot(zscore(df.Timon_B),'LineWidth', 3.5, 'Color',colorBrT)

xlabel('Time (samples)')
ylabel('Value')
legend("Bradykinesia Cleartune", "Bradykinesia Timon")

exportgraphics(gcf, 'Plots/rcs02_L_VerciseDirected/Bradykinesia_Thr700_timeSeries.png', 'Resolution', 300)

%Dyskinesia corrplot
h2=ea_corrplot(df.CT_Amplitude, df.Timon_D, 0, {'', 'CT Amplitude', 'Dyskinesia Timon'}, [], [], colorAmpCT);
h2.Position=[100 100 500 700];
exportgraphics(h2, 'Plots/rcs02_L_VerciseDirected/Amplitude_dyskinesia_corrplot.png', 'Resolution', 300)

% time series for dyskinesia vs amplitude
figure;
set(gcf, 'Units', 'centimeters', 'Position', [5 5 55 15])
plot(zscore(df.CT_Amplitude),'LineWidth', 2.5, 'Color',colorAmpCT)
hold on;
plot(zscore(df.Timon_D),'LineWidth', 3.5, 'Color',colorDyT)

xlabel('Time (samples)')
ylabel('Value')
legend("Amplitude Cleartune", "Dyskinesia Timon")

exportgraphics(gcf, 'Plots/rcs02_L_VerciseDirected/Amplitude_dyskinesia_timeSeries.png', 'Resolution', 300)
