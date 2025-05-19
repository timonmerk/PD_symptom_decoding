from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import os
import numpy as np
from py_neuromodulation import nm_stats

PATH_PER = "publication_figures"
PATH_FIGURES = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf/figures_paper/figures_final'
df_comp = pd.read_csv(os.path.join(PATH_PER, "df_comp_beta_ml.csv"))

# patients_without_peak = ["rcs05r", "rcs07r", "rcs12l", "rcs18l", "rcs20l"] short
patients_without_peak = [ "rcs08l", "rcs10l", "rcs12r", "rcs18l"]
order_ = ["pkg_bk", "pkg_tremor", "pkg_dk"]

df_plt_ = df_comp.copy()

df_plt_.loc[df_plt_["type"] == "corr_ind", "value"] *= -1

# select patients only with peak
df_stat = df_plt_[~df_plt_["sub"].isin(patients_without_peak)]

p_val_bk = nm_stats.permutationTest_relative(
    df_stat.query("label == 'pkg_bk' and type == 'corr_pr'")["value"].values,
    df_stat.query("label == 'pkg_bk' and type == 'corr_ind'")["value"].values,
    False, None, 5000
)[1]  # 0.0004
p_val_dk = nm_stats.permutationTest_relative(
    df_stat.query("label == 'pkg_dk' and type == 'corr_pr'")["value"].values,
    df_stat.query("label == 'pkg_dk' and type == 'corr_ind'")["value"].values,
    False, None, 5000
)[1]  # inv 0.00
p_val_tr = nm_stats.permutationTest_relative(
    df_stat.query("label == 'pkg_tremor' and type == 'corr_pr'")["value"].values,
    df_stat.query("label == 'pkg_tremor' and type == 'corr_ind'")["value"].values,
    False, None, 5000
)[1]  # 0.0064

def get_significance(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."
    

plt.figure(figsize=(3, 5))
hue_order = ["corr_ind", "corr_pr"]
order = ["pkg_bk", "pkg_tremor", "pkg_dk"]
sns.boxplot(
    data=df_plt_.query("type != 'corr_ind_abs'"),
    x="label", y="value",
    hue="type", palette="viridis",
    boxprops=dict(alpha=.3), color=None,
    showfliers=False, showmeans=True,
    order=order_, hue_order=hue_order
)

# Swarmplot for 'corr_ind' only
sns.swarmplot(
    data=df_plt_.query("type == 'corr_ind'"),
    x="label", y="value",
    hue="type", dodge=True,
    alpha=.5, palette="viridis",
    legend=False, order=order_, hue_order=hue_order
)

# Swarmplot for 'corr_pr', color-coded by 'peak_present'
df_peak = df_plt_.query("type == 'corr_pr'")
palette_peak = {0: "red", 1: "black"}

hue_idx = hue_order.index("corr_pr")
n_hues = len(hue_order)
width = 0.4
dodge_center = width * (hue_idx / (n_hues - 1)) - width / 2

for i, label in enumerate(order_):
    sub = df_peak[df_peak["label"] == label]
    y_vals = sub["value"]
    colors = sub["peak_present"].map(palette_peak)

    # Add small random jitter around dodge_center
    jitter = np.random.normal(loc=0, scale=0.05, size=len(sub))
    x_vals = i + dodge_center + jitter

    plt.scatter(x_vals, y_vals, c=colors, alpha=0.5, s=20, edgecolors="black", linewidth=0.5)

# === Add significance bars === #
# Position settings
y_max = df_plt_["value"].max()
y_offsets = [0.1, 0.1, 0.1]  # distance above max value for each bar

# Define comparisons and p-values
comparisons = [
    (0, p_val_bk),
    (1, p_val_tr),
    (2, p_val_dk)
]

for i, (idx, p) in enumerate(comparisons):
    y = y_max + y_offsets[i]
    plt.plot([idx - 0.2, idx + 0.2], [y, y], color="black", linewidth=1)
    plt.text(idx, y + 0.01, get_significance(p), ha='center', va='bottom', fontsize=12)

plt.ylim(-0.85, 1.1)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig(os.path.join(PATH_FIGURES, "beta_ml_comp_inv.pdf"))
plt.show(block=True)

# compute stats vs zero:
nm_stats.permutationTest(
    df_plt_.query("label == 'pkg_bk' and type == 'corr_pr'")["value"].values,
    np.zeros(df_plt_.query("label == 'pkg_bk' and type == 'corr_pr'")["value"].values.shape[0]),
    False, None, 5000
)  # 0.0

nm_stats.permutationTest(
    df_plt_.query("label == 'pkg_dk' and type == 'corr_pr'")["value"].values,
    np.zeros(df_plt_.query("label == 'pkg_dk' and type == 'corr_pr'")["value"].values.shape[0]),
    False, None, 5000
)  # 0.0

nm_stats.permutationTest(
    df_plt_.query("label == 'pkg_tremor' and type == 'corr_pr'")["value"].values,
    np.zeros(df_plt_.query("label == 'pkg_tremor' and type == 'corr_pr'")["value"].values.shape[0]),
    False, None, 5000
)  # 0.0