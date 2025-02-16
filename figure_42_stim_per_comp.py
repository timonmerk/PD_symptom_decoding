import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm
from sklearn import metrics


PATH_FEATURES = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_normalized_10s_window_length/480"
PATH_FIGURES = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf'
df_all = pd.read_csv(os.path.join(PATH_FEATURES, "all_merged_normed.csv"), index_col=0)
df_all = pd.read_csv('/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_rmap/all_ch_renamed_no_rmap.csv')
df_all["pkg_dt"] = pd.to_datetime(df_all["pkg_dt"]).dt.tz_localize("US/Pacific")
subs = np.sort(df_all["sub"].unique())

df_stim_times = pd.read_csv('/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/stim_ampl_freq.csv')
df_stim_times["pkg_dt"] = pd.to_datetime(df_stim_times["pkg_dt"], utc=True).dt.tz_convert("US/Pacific")
#OUT_FILE = "LOHO_ALL_LABELS_ALL_GROUPS_normed_480.pkl"

import pickle
PATH_PER = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per'
OUT_FILE = "LOHO_ALL_LABELS_ALL_GROUPS_normed_480.pkl"
PATH_FIGURES = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf'
#OUT_FILE = "LOHO_ALL_LABELS_ALL_GROUPS_nonorm.pkl"
PATH_READ = os.path.join(PATH_PER, OUT_FILE)

with open(PATH_READ, "rb") as f:
    d_out = pickle.load(f)

l_ = []
l_stim_times = []
for CLASSIFICATION in [True, False]:
    for pkg_label in ["pkg_dk", "pkg_bk", "pkg_tremor"]:
        
        for sub in subs:
            time = d_out[CLASSIFICATION][pkg_label]["ecog_stn"][sub]["time"]
            pr = d_out[CLASSIFICATION][pkg_label]["ecog_stn"][sub]["pr"]
            tr = d_out[CLASSIFICATION][pkg_label]["ecog_stn"][sub]["y_"]

            stim_sub = df_stim_times[df_stim_times["sub"] == sub]
            stim_times_on = stim_sub[stim_sub["stim_ampl"] > 0]
            stim_times_off = stim_sub[stim_sub["stim_ampl"] == 0]

            if stim_times_on.shape[0] == 0:
                continue

            time_pred = pd.Series(time).dt.tz_localize("US/Pacific")
            times_on = stim_times_on["pkg_dt"]
            times_off = stim_times_off["pkg_dt"]

            # get the indices of time_pred that match the ones in time_pred
            for stim_idx, times__ in enumerate([times_on, times_off]):
                
                idx_match = []
                if stim_idx == 0:
                    stim_type = "on"
                else:
                    stim_type = "off"
                pred_ = []
                true_ = []
                for time_on in times__:
                    idx = np.where(time_pred == pd.Timestamp(time_on))[0]
                    if idx.shape[0] == 0:
                        continue
                    idx_match.append(idx[0])
                    pred_.append(pr[idx[0]])
                    true_.append(tr[idx[0]])
                
                if CLASSIFICATION:
                    per = metrics.balanced_accuracy_score(true_, pred_)
                else:
                    per = np.corrcoef(true_, pred_)[0, 1]
            
                if CLASSIFICATION and pkg_label == "pkg_dk":
                    l_stim_times.append({"sub": sub, "stim_type": stim_type, "samples": len(true_)})

                if per == 0:
                    print()
                if len(pred_) < 15:  # limit to at least half an hour
                    continue
                l_.append({"sub": sub, "pkg_label": pkg_label, "classification": CLASSIFICATION, "per": per, "stim_type": stim_type})  
            
df_per = pd.DataFrame(l_)
df_stim_ratio = pd.DataFrame(l_stim_times)

df_stim_sum = df_stim_ratio.groupby(["sub", "stim_type"]).sum().reset_index()
df_stim_sum["sub_unique"] = df_stim_sum["sub"].apply(lambda x: x[:-1])
# 9 unique subjects, 15 hemispheres

# show barplot of the number of samples per subject 
plt.figure(figsize=(5, 5))
sns.barplot(data=df_stim_sum, x="sub", y="samples", hue="stim_type", palette="viridis")
plt.ylabel("Number of samples")
plt.show()

plt.figure(figsize=(5, 7))
plt.subplot(211)
plt.title("Classification")
plt.ylabel("Balanced Accuracy")
sns.boxplot(data=df_per.query("classification == True"), x="pkg_label", y="per", hue="stim_type",
            palette="viridis", boxprops=dict(alpha=.5), order=["pkg_bk", "pkg_dk", "pkg_tremor"], showmeans=True, showfliers=False)
sns.swarmplot(data=df_per.query("classification == True"), x="pkg_label", y="per", hue="stim_type", color=".25", dodge=True, palette="viridis", order=["pkg_bk", "pkg_dk", "pkg_tremor"])
plt.subplot(212)
plt.title("Regression")
plt.ylabel("Correlation coefficient")
sns.boxplot(data=df_per.query("classification == False"), x="pkg_label", y="per", hue="stim_type", palette="viridis",
            boxprops=dict(alpha=.5), order=["pkg_bk", "pkg_dk", "pkg_tremor"], showmeans=True, showfliers=False)
sns.swarmplot(data=df_per.query("classification == False"), x="pkg_label", y="per", hue="stim_type", color=".25", dodge=True, palette="viridis", order=["pkg_bk", "pkg_dk", "pkg_tremor"])
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "figure_42_stim_per_comp.pdf"), dpi=300)
plt.show(block=True)

