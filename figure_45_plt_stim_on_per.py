import os
import pickle
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt

PATH_PER = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per'
SAVE_NAME = "LOHO_ALL_LABELS_ALL_GROUPS_STIM_ON_ONLY.pkl"

with open(os.path.join(PATH_PER, SAVE_NAME), "rb") as f:
    d_out = pickle.load(f)

l = []
for CLASSIFICATION in d_out.keys():
    for pkg_label in d_out[CLASSIFICATION].keys():
        for sub in d_out[CLASSIFICATION][pkg_label]["ecog_stn"].keys():
            if CLASSIFICATION is True:
                per_ = "ba"
            else:
                per_ = "corr_coeff"
            l.append({
                "sub": sub,
                "pkg_label": pkg_label,
                "CLASSIFICATION": CLASSIFICATION,
                "per": d_out[CLASSIFICATION][pkg_label]["ecog_stn"][sub][per_]
            })
df_loso = pd.DataFrame(l)

order_ = ["pkg_bk", "pkg_dk", "pkg_tremor"]

# THOSE RESULTS HAVE TO BE COMPARED TO STIM_OFF PERFORMANCES

plt.figure(figsize=(10, 5))
plt.subplot(121)
sns.boxplot(x="pkg_label", y="per", data=df_loso.query("CLASSIFICATION == True"), showfliers=False, showmeans=True, palette="viridis", boxprops=dict(alpha=0.5), order=order_)
sns.swarmplot(x="pkg_label", y="per", data=df_loso.query("CLASSIFICATION == True"), color=".25", alpha=0.5, palette="viridis", order=order_)
plt.title("Balanced accuracy")
plt.subplot(122)
sns.boxplot(x="pkg_label", y="per", data=df_loso.query("CLASSIFICATION == False"), showfliers=False, showmeans=True, palette="viridis", boxprops=dict(alpha=0.5), order=order_)
sns.swarmplot(x="pkg_label", y="per", data=df_loso.query("CLASSIFICATION == False"), color=".25", alpha=0.5, palette="viridis", order=order_)
plt.title("Correlation coefficient")
plt.show(block=True)
