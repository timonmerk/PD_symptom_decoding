addpath("/Users/Timon/Documents/MATLAB/spm12");
addpath(genpath("/Users/Timon/Documents/MATLAB/leaddbs"));
addpath("/Users/Timon/Documents/MATLAB/wjn_toolbox");
addpath("matlab_funcs");
PATH_PER = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/paper_per/without_night/ind_ch';
df = readtable(fullfile(PATH_PER, 'df_per_ind_all_coords.csv'), 'ReadRowNames', true);

% colors = colorlover(5);
% wjn_plot_surface(fullfile('meshes', 'cortex_bl.nii'), "#FF8000", 1);  % 0, 

% wjn_plot_surface(fullfile('meshes', 'STN_bl.nii'), "#FF8000", 1);  % 0, 
% wjn_plot_surface("'/Users/Timon/Documents/MATLAB/leaddbs/templates/space/MNI152NLin2009bAsym/atlases/DISTAL (Ewert 2017)/lh/GPi.nii'", "#FF8000", 0.3);  % 0, 

fig = ea_mnifigure;
df_query = df((strcmp(df.loc, 'STN') | strcmp(df.loc, 'GP')) & strcmp(df.label, 'pkg_dk') & strcmp(df.classification, 'False'), :);
[contact_colours, colours_rgb] = map_values_to_cmap([df_query.per; 0.0; 0.75], 'viridis');
contact_colours = contact_colours(1:end-2, :);
radius = 0.3;
[x, y, z] = sphere(100);
for row=1:height(df_query)
    %coords = str2num(df_query.ch_coords{row}) * 1000;
    ax_sphere = surf(abs(df_query(row,:).x) + (x.*radius), ...
                     df_query(row,:).y + (y.*radius), ...
                     df_query(row,:).z + (z.*radius));
    set(ax_sphere, 'LineStyle', 'none', 'facecolor', contact_colours(row, :), 'facealpha', 1);
end

df_query = df((strcmp(df.loc, 'STN') | strcmp(df.loc, 'GP')) & strcmp(df.label, 'pkg_tremor') & strcmp(df.classification, 'False'), :);
[contact_colours, colours_rgb] = map_values_to_cmap([df_query.per; 0.0; 0.75], 'viridis');
contact_colours = contact_colours(1:end-2, :);
radius = 0.3;
[x, y, z] = sphere(100);
for row=1:height(df_query)
    %coords = str2num(df_query.ch_coords{row}) * 1000;
    ax_sphere = surf(abs(df_query(row,:).x) + (x.*radius), ...
                     df_query(row,:).y + (y.*radius), ...
                     df_query(row,:).z + (z.*radius));
    set(ax_sphere, 'LineStyle', 'none', 'facecolor', contact_colours(row, :), 'facealpha', 1);
end

df_query = df((strcmp(df.loc, 'STN') | strcmp(df.loc, 'GP')) & strcmp(df.label, 'pkg_bk') & strcmp(df.classification, 'False'), :);
[contact_colours, colours_rgb] = map_values_to_cmap([df_query.per; 0.1; 0.9], 'viridis');
contact_colours = contact_colours(1:end-2, :);
radius = 0.3;
[x, y, z] = sphere(100);
for row=1:height(df_query)
    %coords = str2num(df_query.ch_coords{row}) * 1000;
    ax_sphere = surf(abs(df_query(row,:).x) + (x.*radius), ...
                     df_query(row,:).y + (y.*radius), ...
                     df_query(row,:).z + (z.*radius));
    set(ax_sphere, 'LineStyle', 'none', 'facecolor', contact_colours(row, :), 'facealpha', 1);
end


%%%%%%%%%% ECoG 
df_query = df(strcmp(df.loc, 'ECOG') & strcmp(df.label, 'pkg_bk') & strcmp(df.classification, 'True'), :);
[contact_colours, colours_rgb] = map_values_to_cmap([df_query.per; 0.3; 0.92], 'viridis');
contact_colours = contact_colours(1:end-2, :);

radius = 1.5;
fig = ea_mnifigure;
%[x, y, z] = sphere(100);
for row=1:height(df_query)
    %coords = str2num(df_query.ch_coords{row}) * 1000;
    ax_sphere = surf(abs(df_query(row,:).x) + (x.*radius), ...
                     df_query(row,:).y + (y.*radius), ...
                     df_query(row,:).z + (z.*radius));
    set(ax_sphere, 'LineStyle', 'none', 'facecolor', contact_colours(row, :), 'facealpha', 1);
end

cortex_mesh = load(fullfile('meshes/CortexHiRes.mat'));
plot_mesh.vertices = cortex_mesh.Vertices_rh;
plot_mesh.faces = cortex_mesh.Faces_rh;
[p,s,v] = wjn_plot_surface(plot_mesh,  "#dbdad7", 0.);  % #FF8000
alpha(p, 0.4)
set(gcf,'Color','white')