import pandas as pd
import os
import numpy as np

df_all = pd.read_csv('/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_rmap/all_ch_renamed_no_rmap.csv')

PATH_STIM = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/stimulation_settings.csv'
PATH_OUT = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per'

df_stim = pd.read_csv(PATH_STIM)
# convert the timestamp (in unix format) to datetime
df_stim["pkg_dt"] = pd.to_datetime(df_stim["timestamp"], unit="ms").dt.tz_localize("UTC").dt.tz_convert("US/Pacific")
df_stim.to_csv("stim_with_dt.csv")

df_all["pkg_dt"] = pd.to_datetime(df_all["pkg_dt"]).dt.tz_localize("US/Pacific")

subs = df_all["sub"].unique()

df_stim_out = []

for sub in subs:
    df_sub = df_all[df_all["sub"] == sub]
    df_stim_patient = df_stim[np.logical_and(df_stim["subject_id"] == sub[:-1], df_stim["hemisphere_id"].apply(lambda x: x[0]) == sub[-1])].reset_index(drop=True)
    df_stim_patient = df_stim_patient.sort_values("pkg_dt")
    stim_freq = df_stim_patient["stimulation_frequency"].values
    stim_ampl = df_stim_patient["stimulation_amplitude"].values
    stim_times_start = df_stim_patient["pkg_dt"][:-1].reset_index(drop=True)
    stim_times_stop = df_stim_patient["pkg_dt"][1:].reset_index(drop=True)
    idx_stim = []
    stim_ampl_arr = np.zeros(df_sub.shape[0])
    stim_freq_arr = np.zeros(df_sub.shape[0])
    for i in range(len(stim_freq)-1):
        if stim_freq[i] > 0:
            # get range of stimulation
            x_start = stim_times_start[i]
            x_stop = stim_times_stop[i]
            idx_plt = np.where((df_sub["pkg_dt"] > x_start) & (df_sub["pkg_dt"] < x_stop))[0]
            
            if idx_plt.shape[0] > 0:
                stim_ampl_arr[idx_plt] = stim_ampl[i]
                stim_freq_arr[idx_plt] = stim_freq[i]
                for j in range(idx_plt.shape[0]):
                    idx_stim.append(idx_plt[j])
            
    # highlight last stimulation
    if stim_freq[-1] > 0:
        x_start = stim_times_start.iloc[-1]
        idx_plt = np.where(df_sub["pkg_dt"] > x_start)[0]
        if idx_plt.shape[0] > 0:
            stim_ampl_arr[idx_plt] = stim_ampl[-1]
            stim_freq_arr[idx_plt] = stim_freq[-1]

    df_sub["stim_ampl"] = stim_ampl_arr
    df_sub["stim_freq"] = stim_freq_arr

    df_stim_out.append(df_sub[['pkg_dt', 'stim_ampl', 'stim_freq', 'sub']])

df_stim_out = pd.concat(df_stim_out)
df_stim_out["pkg_dt"] = df_stim_out["pkg_dt"].dt.tz_convert("UTC")
df_stim_out.to_csv(os.path.join(PATH_OUT, "stim_ampl_freq.csv"))