import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

from py_neuromodulation import nm_stats

PATH_FIGURES = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf/figures_paper/figures_final'

# export minus sign matplotlib
plt.rcParams["axes.unicode_minus"] = False
# set font size to 5 and font family to Arial
plt.rcParams.update({"font.size": 5, "font.family": "Arial"})

PATH_PER = "publication_figures"
LOCATIONS = ["SC", "ECOG", "ECOG_SC"]
LABELS = ["pkg_bk", "pkg_tremor", "pkg_dk"]


def get_significance(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "n.s."


df_STN_ML = pd.read_csv(f"{PATH_PER}/df_comp_beta_ml_stn.csv").query("type == 'corr_pr'")
df_ECOG_ML = pd.read_csv(f"{PATH_PER}/df_comp_beta_ml_ecog.csv").query("type == 'corr_pr'")
df_ECOGSTN_ML = pd.read_csv(f"{PATH_PER}/df_comp_beta_ml_ecog_stn.csv").query("type == 'corr_pr'")
df_STN_ML["location"] = "SC"
df_ECOG_ML["location"] = "ECOG"
df_ECOGSTN_ML["location"] = "ECOG_SC"
df = pd.concat([df_STN_ML, df_ECOG_ML, df_ECOGSTN_ML], axis=0)

for label in LABELS:
    # compare by relative state STN, ECOG, ECOG+STN
    df_label = df.query("label == @label")
    for loc in LOCATIONS:
        for loc2 in LOCATIONS:
            if loc != loc2:
                pval = nm_stats.permutationTest_relative(
                    df_label.query("location == @loc")["value"].values,
                    df_label.query("location == @loc2")["value"].values,
                    False, None, 5000,
                )[1]
                print(f"{label} {loc} vs {loc2}: p={pval:.4f}")

plt.figure(figsize=(3, 2))
ax = sns.boxplot(
    data=df,
    x="label",
    y="value",
    hue="location",
    order=LABELS,
    hue_order=LOCATIONS,
    palette="viridis",
    showmeans=True,
    showfliers=False,
    boxprops=dict(alpha=0.7),
)
sns.swarmplot(
    data=df,
    x="label",
    y="value",
    hue="location",
    order=LABELS,
    hue_order=LOCATIONS,
    dodge=True,
    color=".25",
    alpha=0.5,
    size=1,
    legend=False,
)

# Add within-group significance comparisons between the location boxes.
label_offsets = np.linspace(-0.25, 0.25, len(LOCATIONS))
max_y_for_annotations = df["value"].max()
for label_idx, label in enumerate(LABELS):
    df_label = df.query("label == @label")
    for i, loc in enumerate(LOCATIONS):
        for j in range(i + 1, len(LOCATIONS)):
            loc2 = LOCATIONS[j]
            x1 = label_idx + label_offsets[i]
            x2 = label_idx + label_offsets[j]
            vals1 = df_label.query("location == @loc")["value"].values
            vals2 = df_label.query("location == @loc2")["value"].values
            _, pval = nm_stats.permutationTest_relative(vals1, vals2, False, None, 5000)
            sig = get_significance(pval)
            y_bar = max(vals1.max(), vals2.max()) + 0.02
            bar_height = 0.05
            ax.plot(
                [x1, x1, x2, x2],
                [y_bar, y_bar + bar_height, y_bar + bar_height, y_bar],
                color="black",
                linewidth=1.2,
            )
            ax.text(
                (x1 + x2) / 2,
                y_bar + bar_height + 0.015,
                sig,
                ha="center",
                va="bottom",
                fontsize=5,
                color="black",
            )
            max_y_for_annotations = max(max_y_for_annotations, y_bar + bar_height + 0.08)

sns.despine()
plt.title("Comparison of ML performance by location")
plt.xlabel("")
plt.ylabel("Pearson Correlation coefficient")
plt.legend(title="Location")
plt.ylim(bottom=min(df["value"].min() - 0.05, -0.05), top=max_y_for_annotations * 1.1)
plt.savefig(os.path.join(PATH_FIGURES, "figure_54_compare_ML_STN_ECOG.pdf"))


# pkg_bk SC vs ECOG: p=0.4800
# pkg_bk SC vs ECOG_SC: p=0.0030
# pkg_bk ECOG vs SC: p=0.4766
# pkg_bk ECOG vs ECOG_SC: p=0.0174
# pkg_bk ECOG_SC vs SC: p=0.0026
# pkg_bk ECOG_SC vs ECOG: p=0.0204
# pkg_dk SC vs ECOG: p=0.5846
# pkg_dk SC vs ECOG_SC: p=0.5926
# pkg_dk ECOG vs SC: p=0.5806
# pkg_dk ECOG vs ECOG_SC: p=0.0610
# pkg_dk ECOG_SC vs SC: p=0.5918
# pkg_dk ECOG_SC vs ECOG: p=0.0630
# pkg_tremor SC vs ECOG: p=0.9458
# pkg_tremor SC vs ECOG_SC: p=0.0082
# pkg_tremor ECOG vs SC: p=0.9424
# pkg_tremor ECOG vs ECOG_SC: p=0.0946
# pkg_tremor ECOG_SC vs SC: p=0.0078
# pkg_tremor ECOG_SC vs ECOG: p=0.0914
# <string>:58: FutureWarning: 