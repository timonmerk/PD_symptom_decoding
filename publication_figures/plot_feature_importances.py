import pickle
import os
import numpy as np
import pandas as pd
import re
from matplotlib import pyplot as plt
import seaborn as sns

PATH_FIGURES = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf/figures_paper/figures_final'
PATH_OUT = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/final_models"
file_name = os.path.join(PATH_OUT, f"feature_importances_cb.pkl")

with open(file_name, "rb") as f:
    feature_importances = pickle.load(f)

# there are Fooog, Hjorth, PSD in different bands, fft in different bands, sharpwave 
feature_groups = ["Hjorth", "fooof", "Sharpwave", "theta", "alpha", "low beta", "high beta", "low gamma", "high gamma", "hour", "raw", "LineLength"]

frequency_ranges = {
    "delta": [0, 4],
    "theta": [4, 8],
    "alpha": [8, 12],
    "low beta": [12, 20],
    "high beta": [20, 35],
    "low gamma": [35, 80],
    "high gamma": [80, 200],
}

# if psd in feature name, check the range; 'ch_subcortex_welch_psd_5_std' -> 'welch_psd_5' -> 5 Hz -> 'theta'
def get_feature_group(feature_name):
    for f in feature_groups:
        if f in feature_name:
            return f
    if "psd" in feature_name:
        match = re.search(r'psd_(\d+)', feature_name)
        if match:
            freq = int(match.group(1))
            for band, (low, high) in frequency_ranges.items():
                if low <= freq < high:
                    return band
    return "Other"

groups_ = []
for label in ["pkg_bk", "pkg_dk", "pkg_tremor"]:
    for loc in ["ecog", "stn", "ecog_stn"]:
        feature_importances_ = feature_importances[label][loc]
        feature_names = feature_importances_["feature_names"]
        importances = feature_importances_["feature_importances"]
        groups = [get_feature_group(fn) for fn in feature_names]
        # average importances per group
        df_imp = pd.DataFrame({"feature": feature_names, "importance": importances, "group": groups})
        df_grouped = df_imp.groupby("group")["importance"].sum().reset_index()
        df_grouped["loc"] = loc
        df_grouped["label"] = label
        groups_.append(df_grouped)

# Combine all groups into a single DataFrame
df_all = pd.concat(groups_, ignore_index=True)

oder_ = ["delta", "theta", "alpha", "low beta", "high beta", "low gamma", "high gamma", "raw", "LineLength", "Hjorth", "fooof", "Sharpwave", "hour",]

# plot using seaborn
plt.figure(figsize=(12, 6))
for i, label in enumerate(["pkg_bk", "pkg_tremor", "pkg_dk"]):
    plt.subplot(1, 3, i+1)
    sns.barplot(data=df_all[df_all["label"] == label],
                x="importance", y="group", hue="loc", palette="viridis", order=oder_)
    plt.title(f"{label}")
    plt.xlabel("CatBoost Sum Feature Importances")
    plt.ylabel("Feature Group")
    if i == 0:
        plt.yticks(ticks=np.arange(len(oder_)), labels=oder_)
    else:
        plt.yticks(ticks=np.arange(len(oder_)), labels=[""] * len(oder_))

plt.legend(title="Location")
plt.suptitle("Sum Feature Importances by Group for CatBoost Models")
plt.savefig(os.path.join(PATH_FIGURES, "figure_feature_importances_sum.pdf"))
