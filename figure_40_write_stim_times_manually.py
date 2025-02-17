import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm

PATH_OUT = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per'
PATH_FEATURES = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/"
PATH_FIGURES = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf'
df_all = pd.read_csv('/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_rmap/all_ch_renamed_no_rmap.csv')
df_all["pkg_dt"] = pd.to_datetime(df_all["pkg_dt"]).dt.tz_localize("US/Pacific")
subs = np.sort(df_all["sub"].unique())

df_stim_times = pd.read_csv('/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/stim_ampl_freq.csv')
df_stim_times["pkg_dt"] = pd.to_datetime(df_stim_times["pkg_dt"], utc=True).dt.tz_convert("US/Pacific")
#OUT_FILE = "LOHO_ALL_LABELS_ALL_GROUPS_normed_480.pkl"

ind_peaks = {
    "rcs02l" : 20,
    "rcs02r" : 18,
    "rcs03l" : 13.5,
    "rcs05l" : 25,
    "rcs05r" : 25,
    "rcs06l" : 20,
    "rcs06r" : 18,
    "rcs07l" : 20,
    "rcs07r" : 20,
    "rcs08l" : 20,
    "rcs08r" : 20,
    "rcs09l" : 23,
    "rcs09r" : 23,
    "rcs10l" : 27,
    "rcs10r" : 30,
    "rcs11l" : 25,
    "rcs11r" : 17,
    "rcs12l" : 27,
    "rcs12r" : 20,
    "rcs14l" : 25,
    "rcs15l" : 18,
    "rcs15r" : 18,
    "rcs17l" : 25,
    "rcs17r" : 27,
    "rcs18l" : 23,
    "rcs18r" : 23,
    "rcs19l" : 22,
    "rcs19r" : 25,
    "rcs20l" : 18,
    "rcs20r" : 17,
}

stim_ranges = {
    "02l": "241",
    "03l": "0,1, 25-30", #(has line noise artifacts)
    "07r": "560",
    "08r": "76, 7-8, 0-3",
    "09l": "79, 104, 107-110", # (CHECK)
    "09r": "347, 375, 376",
    "10l": "16-18, 87-527, 571, 573",
    "10r": "19-21, 93-361, 380-381",
    "11l": "333-341, 361, 367-373, 375-408",
    "11r": "90, 205-303",
    "12l": "478-560",# (CHECK again, correct)
    "12r": "386-391",
    #"14l": (has artefacts)
    "15r": "0-39, 309-717",
    "17l": "377-516",
    "17r": "404-542",
    "18l": "618-701",
    "18r": "550-633",
}

stim_ranges_all_times = {
    "02l": "277, 238, 239",
    "03l": "0,1, 25-30", #(has line noise artifacts)
    "07r": "1143",
    "08l": "251",
    "08r": "10-11, 123-126, 197-198",
    "09l": "104, 132-135", # (CHECK)
    "09r": "389, 417, 418",
    "10l": "16-18, 87-638, 682, 757",
    "10r": "19-21, 93-94, 99-428, 447-448",
    "11l": "414-415, 474-572, 545-578",
    "11r": "473-572",
    #"12l": "478-560",# (CHECK again, correct)
    "12r": "748-753",
    #"14l": (has artefacts)
    "15r": "0-39, 309-973",
    "17l": "377-516",
    "17r": "404-542",
    "18l": "697-778",
    "18r": "623-704",
}

artifacts_annotate = {
    "03l": "0,1, 25-30",
    "09l": "104, 132-135",
    "09r": "389, 417, 418",
    "10l": "87-89, 158-160",
    "10r": "99-100, 174-175",
}

def expand_stim_ranges(stim_ranges):
    expanded_dict = {}
    
    for key, value in stim_ranges.items():
        # Remove comments if present
        value = value.split('#')[0].strip()
        
        individual_entries = []
        
        # Split by commas and process each entry
        for entry in value.split(','):
            entry = entry.strip()
            
            if '-' in entry:  # Handle ranges
                start, end = map(int, entry.split('-'))
                individual_entries.extend(range(start, end + 1))
            else:  # Handle single values
                individual_entries.append(int(entry))
        
        expanded_dict[key] = sorted(individual_entries)  # Store sorted list in dict
    
    return expanded_dict

expanded_dict_stim = expand_stim_ranges(stim_ranges_all_times)
expanded_dict_artifacts = expand_stim_ranges(artifacts_annotate)

df_all["h"] = df_all["pkg_dt"].dt.hour
# Plot example prediction beta vs decoding

stim_times_sub = {}
artifact_times_sub = {}
times_use_stim_off = {}

df_subs = []
for sub in np.sort(subs):
    mean_spectra_sub = []
    print(sub)
    #sub = "rcs05l"
    df_sub = df_all[df_all["sub"] == sub].reset_index(drop=True)

    df_psd = df_sub[[f'ch_subcortex_1_welch_psd_{i}_mean' for i in range(126)]]
    df_psd["pkg_dt"] = df_sub["pkg_dt"]
    df_psd["h"] = df_sub["h"]
    # remove nan rows
    df_psd = df_psd.dropna()
    idx_not_nan = df_psd.index
    df_sub = df_sub.iloc[idx_not_nan].reset_index(drop=True)

    times_stim_sub = []
    if sub[3:] in expanded_dict_stim:
        times_idx = expanded_dict_stim[sub[3:]]
        times_stim_sub = df_sub["pkg_dt"].iloc[times_idx]
    stim_times_sub[sub] = times_stim_sub

    times_artifact_sub = []
    if sub[3:] in expanded_dict_artifacts:
        times_idx = expanded_dict_artifacts[sub[3:]]
        times_artifact_sub = df_sub["pkg_dt"].iloc[times_idx]
    artifact_times_sub[sub] = times_artifact_sub

    # rest of the time is stim_off
    times_use_stim_off = df_sub[~df_sub["pkg_dt"].isin(times_stim_sub)]
    times_use_stim_off = times_use_stim_off[~times_use_stim_off["pkg_dt"].isin(times_artifact_sub)]
    times_use_stim_off = times_use_stim_off["pkg_dt"]
    times_use_stim_off = times_use_stim_off.reset_index(drop=True)

    # set df_all column, set Artifact if in times_artifact_sub, set stim if in times_stim_sub, else stim_off
    df_sub["stim"] = False
    df_sub["artifact"] = False
    df_sub["stim_off"] = False

    for time in times_stim_sub:
        idx = df_sub[df_sub["pkg_dt"] == time].index[0]
        df_sub.at[idx, "stim"] = True
    for time in times_artifact_sub:
        idx = df_sub[df_sub["pkg_dt"] == time].index[0]
        df_sub.at[idx, "artifact"] = True
    for time in times_use_stim_off:
        idx = df_sub[df_sub["pkg_dt"] == time].index[0]
        df_sub.at[idx, "stim_off"] = True
    df_sub["condition"] = df_sub.apply(lambda row: "artifact" if row["artifact"] else "stim" if row["stim"] else "stim_off", axis=1)
    df_subs.append(df_sub)

df_all_ = pd.concat(df_subs).to_csv(os.path.join(PATH_FEATURES, "all_artifact_and_stim_condition.csv"))

# save output to pickle
import pickle
with open(os.path.join(PATH_OUT, "stim_times_subs.pkl"), "wb") as f:
    pickle.dump(stim_times_sub, f)

with open(os.path.join(PATH_OUT, "artifact_times_subs.pkl"), "wb") as f:
    pickle.dump(artifact_times_sub, f)

