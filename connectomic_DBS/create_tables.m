%% Create tables from the Cleartune optimization results
% By Patricia (October 16, 2025)

clc
clear

cd '/Volumes/netstim/PROJECTS/patriciazvarova/DynamicCleartune/Timon_results_2share/'
% load the downsampled table with Timon's results
load('Raw_results/df_out_predictions_true_norm_downsampled_CORRECT.mat');

% define varaible names for the table
table_names = {'Timon_T' % tremor predictions by Timon
    'Timon_B' % bradykinesia predictions by Timon
    'Timon_D' % Dyskinesia predictions by Timon
    'CT_T' % Ratio of fiber count for tremor if ClearTune settings
    'CT_B' % Ratop of fiber count for bradykinesia if ClearTune settings
    'Target_Amplitude' % Target Amplitude adjusted by Timon's dyskinesia
    'CT_Amplitude'}; % Amplitude of the winning settings by CT

%% rcs02 left hemisphere

% optimizer file results
load('Raw_results/sub-rcs02_desc-optimizerstatus_hemi-L.mat');

% load the fiber counts count data
thr500=load('Raw_results/rcs02_L_fiberCountsConnected_thr500.mat');

%only work with the first patient left hemisphere here
this_pat_data=normValuesTable{1};

% also get the target amplitude adjusted for dyskinesia value
standardAmplitude=3;
DyskinesiaVal=table2array(this_pat_data(:,4));
dyskinesia_adjusted_amplitude=standardAmplitude-(DyskinesiaVal-0.5);

% get the fiber counts for bradylinesia and tremor separately
bradykinesa_ct_thr500 = zeros(1, length(thr500.allConnectedCounts));
tremor_ct_thr500 = zeros(1, length(thr500.allConnectedCounts));

for i = 1:length(thr500.allConnectedCounts)
    bradykinesa_ct_thr500(i) = thr500.allConnectedCounts{i}{1,2};
    tremor_ct_thr500(i) = thr500.allConnectedCounts{i}{2,2};
end

% calculate the % ratio between bradykinesia and tremor
Br = bradykinesa_ct_thr500./(bradykinesa_ct_thr500+tremor_ct_thr500)*100;
Tr = tremor_ct_thr500./(bradykinesa_ct_thr500+tremor_ct_thr500)*100;

%sanity check
checkPerc=all(Br + Tr == 100);

data = [this_pat_data.zs_norm2one_pkg_tremor'; this_pat_data.zs_norm2one_pkg_bk'; this_pat_data.zs_norm2one_pkg_dk'; Tr; Br; dyskinesia_adjusted_amplitude'; abs(sum(ipL.X'))];

data_t = data';  
df = array2table(data_t, 'VariableNames', table_names);

% save('Organized_results/df_rcs02_left.mat','df')

clear df
clear data
%% rcs02 right hemisphere

% load the optimizer file
load('Raw_results/sub-rcs02_desc-optimizerstatus_hemi-R.mat');

% load the fiber count file
thr500 = load('Raw_results/rcs02_R_fiberCountsConnected_thr500.mat');

% taking the right side for the patient
this_pat_data=normValuesTable{2};
standardAmplitude=3;

% also get the target amplitude adjusted for dyskinesia value
DyskinesiaVal=table2array(this_pat_data(:,4));
dyskinesia_adjusted_amplitude=standardAmplitude-(DyskinesiaVal-0.5);

% get the fiber counts for bradylinesia and tremor separately
bradykinesa_ct_thr500 = zeros(1, length(thr500.allConnectedCounts));
tremor_ct_thr500 = zeros(1, length(thr500.allConnectedCounts));

for i = 1:length(thr500.allConnectedCounts)
    bradykinesa_ct_thr500(i) = thr500.allConnectedCounts{i}{1,1};
    tremor_ct_thr500(i) = thr500.allConnectedCounts{i}{2,1};
end

% calculate the % ratio between bradykinesia and tremor
Br = bradykinesa_ct_thr500./(bradykinesa_ct_thr500+tremor_ct_thr500)*100;
Tr = tremor_ct_thr500./(bradykinesa_ct_thr500+tremor_ct_thr500)*100;

%sanity check
checkPerc=all(Br + Tr == 100);

data = [this_pat_data.zs_norm2one_pkg_tremor';this_pat_data.zs_norm2one_pkg_bk'; this_pat_data.zs_norm2one_pkg_dk'; Tr; Br; dyskinesia_adjusted_amplitude'; abs(sum(ipR.X'))];

data_t = data';  
df = array2table(data_t, 'VariableNames', table_names);

save('Organized_results/df_rcs02_right.mat','df')

clear df
clear data

%% Test with Vercise Directed lead 
% cartesia for now isn't supported, results also not what we expected 

load('Raw_results/sub-rcs02VerciseDirect_desc-optimizerstatus_hemi-L.mat')

%get the left hemisphere data
this_pat_data=normValuesTable{1};

thr500 = load('Raw_results/rcs02VerciseDirect_L_fiberCountsConnected_thr500.mat');
thr700 = load('Raw_results/rcs02VerciseDirect_L_fiberCountsConnected_thr700.mat');

% get the fiber counts for bradylinesia and tremor separately
bradykinesa_ct_thr500 = zeros(1, length(thr500.allConnectedCounts));
tremor_ct_thr500 = zeros(1, length(thr500.allConnectedCounts));

bradykinesa_ct_thr700 = zeros(1, length(thr700.allConnectedCounts));
tremor_ct_thr700 = zeros(1, length(thr700.allConnectedCounts));

for i = 1:length(thr500.allConnectedCounts)
    bradykinesa_ct_thr500(i) = thr500.allConnectedCounts{i}{1,2};
    tremor_ct_thr500(i) = thr500.allConnectedCounts{i}{2,2};

    bradykinesa_ct_thr700(i) = thr700.allConnectedCounts{i}{1,2};
    tremor_ct_thr700(i) = thr700.allConnectedCounts{i}{2,2};
end

% calculate the % ratio between bradykinesia and tremor
Br = bradykinesa_ct_thr500./(bradykinesa_ct_thr500+tremor_ct_thr500)*100;
Tr = tremor_ct_thr500./(bradykinesa_ct_thr500+tremor_ct_thr500)*100;

Br700 = bradykinesa_ct_thr700./(bradykinesa_ct_thr700+tremor_ct_thr700)*100;
Tr700 = tremor_ct_thr700./(bradykinesa_ct_thr700+tremor_ct_thr700)*100;

%sanity check
checkPerc=all(Br700 + Tr700 == 100);

data = [this_pat_data.zs_norm2one_pkg_tremor';this_pat_data.zs_norm2one_pkg_bk'; this_pat_data.zs_norm2one_pkg_dk'; Tr; Br; Tr700; Br700; abs(sum(ipL.X'))];

table_names_adjusted = {'Timon_T' % tremor predictions by Timon
    'Timon_B' % bradykinesia predictions by Timon
    'Timon_D'
    'CT_T' % Ratio of fiber count for tremor if ClearTune settings
    'CT_B' % Ratio of fiber count for bradykinesia if ClearTune settings
    'CT_T_Thr700' % Ratio of fiber count for tremor if VTA threshold = 700
    'CT_B_Thr700' % Ratio of fiber count for bradykinesia if VTA threshold = 700
    'CT_Amplitude'}; 

data_t = data';  
df = array2table(data_t, 'VariableNames', table_names_adjusted);

save('Organized_results/df_rcs02VerciseDirected_left.mat','df')

clear df
clear data





