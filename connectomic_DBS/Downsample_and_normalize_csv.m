%% Downsample and rnomalize .csv
% By Patricia August 18 2025

cd '/Volumes/Fornix/Patricia/Timon/Timon_results_2share/'

df=readtable('Raw_results/df_out_predictions_true_class_zs.csv');

% find individual patients
uniqueVals = unique(df.sub);

%loop through the list of unique values to find indices and downsample the
%time points by 6
downSampleVal=6;

for patient = 1:length(uniqueVals)
    idx{patient} = find(strcmp(df.sub, uniqueVals{patient}));
    patientSpecificDf = df(idx{patient},:);
    
    %select which values to average
    vals = [patientSpecificDf.pr_zs_pkg_bk, ...
        patientSpecificDf.pr_zs_pkg_tremor, ...
        patientSpecificDf.pr_zs_pkg_dk];
    
     % truncate to a multiple of 6
    totalRows = size(vals, 1);
    fullBinCount = floor(totalRows / downSampleVal);
    remainder = mod(totalRows, downSampleVal);
    
    % First, process full bins
    fullBinVals = vals(1:fullBinCount * downSampleVal, :);
    reshapedVals = reshape(fullBinVals, downSampleVal, [], 3);  % [6 x numBins x 3]
    avgVals_full = squeeze(mean(reshapedVals, 1));              % [numBins x 3]
    if isvector(avgVals_full)
        avgVals_full = reshape(avgVals_full, 1, []);  % Guarantees it’s 1 row with 3 columns
    end
    % Then process remainder if it exists
    if remainder > 0
        leftoverVals = vals(end - remainder + 1:end, :);        % last N rows
        avgVals_remainder = mean(leftoverVals, 1);              % [1 x 3]
        avgVals = [avgVals_full; avgVals_remainder];            % concatenate
    else
        avgVals = avgVals_full;
    end

    avgVals = reshape(avgVals, [], 3);

    %bring all values to be positive and sum up to 1 per each time point -
    %cleartune always normalizes the values
    vals_minmax=ea_minmax(avgVals(:,1:2));
    normalizedVals = vals_minmax ./ sum(vals_minmax,2);

    % normalize dystonia value
    normalizedDysk = ea_minmax(avgVals(:,3));
    % normalizedDysk = dysk_minmax ./ sum(dysk_minmax);
    patient_idx_name = repmat(uniqueVals{patient},length(normalizedVals(:,1)),1);
    normValuesTable{patient} = table(patient_idx_name, normalizedVals(:,1), normalizedVals(:,2), normalizedDysk, ...        
        'VariableNames', {'patient_name','zs_norm2one_pkg_bk', 'zs_norm2one_pkg_tremor', 'zs_norm2one_pkg_dk'});
end


save('Raw_results/df_out_predictions_true_norm_downsampled_CORRECT.mat.mat','normValuesTable','-mat')
