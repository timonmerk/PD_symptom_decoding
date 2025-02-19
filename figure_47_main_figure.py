import pandas as pd

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from py_neuromodulation import nm_stats
from scipy import stats
import seaborn as sns
import pickle
import yaml

def read_file(PATH_READ, CLASSIFICATION, label_name, loc_):
    with open(PATH_READ, "rb") as f:
        d_out = pickle.load(f)
    data = []
    #for CLASSIFICATION in d_out.keys():
    if CLASSIFICATION:
        per_ = "ba"
    else:
        per_ = "corr_coeff"
    #    for pkg_decode_label in d_out[CLASSIFICATION].keys():
    #        for loc in d_out[CLASSIFICATION][pkg_decode_label].keys():
    for sub in d_out.keys():
        data.append({
            "CLASSIFICATION": CLASSIFICATION,
            "per": d_out[sub][per_],
            "sub": sub,
            "pkg_decode_label": label_name,
            "loc": loc_
        })

    df = pd.DataFrame(data)
    return df

PATH_PER = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/paper_per'

l_all = []
for label_name in ["pkg_dk", "pkg_tremor", "pkg_bk"]:
    for CLASSIFICATION in [True, False]:
        for loc_ in ["ecog", "stn", "ecog_stn"]:
            PATH_NAME = f"LOHO_main_{label_name}_CLASS_{CLASSIFICATION}_loc_{loc_}_nonorm_withpsd.pkl"
            PATH_READ = os.path.join(PATH_PER, PATH_NAME)
            df = read_file(PATH_READ, CLASSIFICATION, label_name, loc_)
            l_all.append(df)
df = pd.concat(l_all)

PATH_FIGURES = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf'
#OUT_FILE = "LOHO_ALL_LABELS_ALL_GROUPS_nonorm.pkl"

def set_box_alpha(ax, alpha=0.5):
    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, alpha))

plt.figure(figsize=(3, 8.2))
idx_plt = 0
x_order = ["ecog", "stn", "ecog_stn"]
for label in ["pkg_bk", "pkg_dk", "pkg_tremor"]:
    for CLASSIFICATION in [True, False]:
        idx_plt += 1
        plt.subplot(3, 2, idx_plt)
        ax = sns.boxplot(data=df.query("CLASSIFICATION == @CLASSIFICATION and pkg_decode_label == @label"), x="loc", y="per", palette="viridis", showfliers=False, showmeans=True, order=x_order); set_box_alpha(ax)
        sns.swarmplot(data=df.query("CLASSIFICATION == @CLASSIFICATION and pkg_decode_label == @label"), x="loc", y="per", dodge=False, color=".25", palette="viridis", alpha=0.5, s=3, order=x_order)
        plt.ylabel("Balanced Accuracy" if CLASSIFICATION else "Correlation Coefficient")
        plt.title(f"{label} {'CLASS' if CLASSIFICATION else 'REG'}")
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.xticks([0, 1, 2], ["ECoG", "SC", "ECoG+SC"], rotation=90)
        plt.xlabel("")

#plt.savefig(os.path.join(PATH_FIGURES, "figure_34_per_groups.pdf"))
plt.tight_layout()
plt.show(block=True)