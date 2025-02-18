import pandas as pd
import os

PATH_READ = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_normalized_10s_window_length/480"
PATH_READ_BASE = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_normalized"

df_condition = pd.read_csv('/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/all_artifact_and_stim_condition.csv')
df_condition["pkg_dt"] = pd.to_datetime(df_condition["pkg_dt"], utc=True).dt.tz_convert("US/Pacific")

for norm_window in [0, 5, 10, 20, 30, 60, 120, 180, 300, 480, 720, 960, 1200, 1440]:
    print(norm_window)
    df_features = pd.read_csv(os.path.join(PATH_READ_BASE, str(norm_window), "all_merged_normed.csv"))
    df_features["pkg_dt"] = pd.to_datetime(df_features["pkg_dt"]).dt.tz_localize("US/Pacific")

    # match the to pkg_dt times and add the condition column to df_features
    df_features["condition"] = None
    for idx, row in df_condition.iterrows():
        idx_match = df_features[df_features["pkg_dt"] == row["pkg_dt"]].index
        if idx_match.shape[0] == 0:
            continue
        df_features.loc[idx_match, "condition"] = row["condition"]

    df_features["condition"].unique()
    # drop None rows in condition
    df_features = df_features.dropna(subset=["condition"]).reset_index(drop=True)
    df_features.to_csv(os.path.join(PATH_READ_BASE, str(norm_window), "all_merged_normed_with_condition.csv"))


PLT_ = False
if PLT_:
    from matplotlib import pyplot as plt
    import numpy as np

    sub = "rcs17l"
    df_sub = df_features[df_features["sub"] == sub]
    df_sub["condition"].unique()
    df_psd = df_sub[[f'ch_subcortex_welch_psd_{i}_max' for i in range(126)]]
    df_psd["condition"] = df_sub["condition"]
    df_stim = df_psd[df_psd["condition"] == "stim"]
    df_stimoff = df_psd[df_psd["condition"] == "stim_off"]
    df_artifact = df_psd[df_psd["condition"] == "artifact"]

    plt.figure(figsize=(15, 8))
    plt.subplot(111)
    for cnt, idx in enumerate(df_stim.index):
        plt.plot(np.arange(126), df_stim.iloc[cnt, :-1], color="red", alpha=0.2)
    for cnt, idx in enumerate(df_stimoff.index):
        plt.plot(np.arange(126), df_stimoff.iloc[cnt, :-1], color="blue", alpha=0.02)
    for cnt, idx in enumerate(df_artifact.index):
        plt.plot(np.arange(126), df_artifact.iloc[cnt, :-1], color="black", alpha=0.2)
    plt.show(block=True)
