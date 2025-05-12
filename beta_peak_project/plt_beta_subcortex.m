addpath("/Users/Timon/Documents/MATLAB/spm12");
addpath("/Users/Timon/Documents/MATLAB/leaddbs");
addpath("/Users/Timon/Documents/MATLAB/wjn_toolbox");
addpath("matlab_funcs")
PATH_PER = '/Users/Timon/Documents/py_neuro_ucsf/py_neuromodulation/beta_peak_project/per_combined_coords_beta.csv';
df = readtable(PATH_PER, 'ReadRowNames', true);

% colors = colorlover(5);
% wjn_plot_surface(fullfile('meshes', 'cortex_bl.nii'), "#FF8000", 1);  % 0, 

% wjn_plot_surface(fullfile('meshes', 'STN_bl.nii'), "#FF8000", 1);  % 0, 
% wjn_plot_surface("'/Users/Timon/Documents/MATLAB/leaddbs/templates/space/MNI152NLin2009bAsym/atlases/DISTAL (Ewert 2017)/lh/GPi.nii'", "#FF8000", 0.3);  % 0, 

fig = ea_mnifigure;
% plot only contacts without 
[contact_colours, colours_rgb] = map_values_to_cmap([zeros(size(df.per_ind)); -1; 1], 'viridis');
contact_colours = repmat('#FF0000', 60, 1);
contact_colours = contact_colours(1:end-2, :);
radius = 0.3;
[x, y, z] = sphere(100);
for row=1:height(df)
    %coords = str2num(df_query.ch_coords{row}) * 1000;
    ax_sphere = surf(abs(df(row,:).x) + (x.*radius), ...
                     df(row,:).y + (y.*radius), ...
                     df(row,:).z + (z.*radius));
    set(ax_sphere, 'LineStyle', 'none', 'facecolor', contact_colours(row, :), 'facealpha', 1);
end
% wjn_plot_surface('/Users/Timon/Documents/MATLAB/leaddbs/templates/space/MNI152NLin2009bAsym/atlases/DISTAL (Ewert 2017)/rh/GPi.nii', "#FF8000", 0.1);  % 0, 

% color-code by performances
[contact_colours, colours_rgb] = map_values_to_cmap([df.per_ind; -1; 1], 'viridis');
contact_colours = contact_colours(1:end-2, :);
radius = 0.3;
[x, y, z] = sphere(100);
for row=1:height(df)
    %coords = str2num(df_query.ch_coords{row}) * 1000;
    ax_sphere = surf(abs(df(row,:).x) + (x.*radius), ...
                     df(row,:).y + (y.*radius), ...
                     df(row,:).z + (z.*radius));
    set(ax_sphere, 'LineStyle', 'none', 'facecolor', contact_colours(row, :), 'facealpha', 1);
end

% next plot
[contact_colours, colours_rgb] = map_values_to_cmap([df.per_all; -1; 1], 'viridis');
contact_colours = contact_colours(1:end-2, :);
radius = 0.3;
[x, y, z] = sphere(100);
for row=1:height(df)
    %coords = str2num(df_query.ch_coords{row}) * 1000;
    ax_sphere = surf(abs(df(row,:).x) + (x.*radius), ...
                     df(row,:).y + (y.*radius), ...
                     df(row,:).z + (z.*radius));
    set(ax_sphere, 'LineStyle', 'none', 'facecolor', contact_colours(row, :), 'facealpha', 1);
end
