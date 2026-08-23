import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os
import numpy as np
from scipy import stats
plt.rcParams.update({"font.size": 5, "font.family": "Arial"})

df_scores = pd.read_csv("ClinicalScoresTable.csv")
df_per = pd.read_csv("publication_figures/df_comp_beta_ml_ecog_stn.csv")

FIGURE_PATH = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf/figures_paper/figures_final"

df_per["sub"] = df_per["sub"].str[:5]
df_scores = df_scores.rename(columns={"Study id": "sub"})
df_scores["tremor_total"] = [2, 7, 21, 9, 0, 20, 20, 4, 14, 11, 14, 6, 2, 17, 8, 23]
df_per = df_per.groupby(["sub", "label", "type"]).mean().reset_index()
df_merged = pd.merge(df_per, df_scores, on="sub", how="inner")
df = df_merged.query("type == 'corr_pr'")

score_col_mapping = {
    "pkg_bk": "UPDRS (Off)",
    "pkg_dk": "UPDRS IV",
    "pkg_tremor": "tremor_total",
}

color_mapping = {
    "pkg_bk": "#376EB4",
    "pkg_dk": "#29AF7F",
    "pkg_tremor": "#DF4A4A",
}

plt.figure(figsize=(3, 1.5))
for label_idx, label in enumerate(["pkg_bk", "pkg_dk", "pkg_tremor"]):
    plt.subplot(1, 3, label_idx + 1)
    # get stats
    r, p = stats.pearsonr(
        df.query("label == @label")["value"].values,
        df.query("label == @label")[score_col_mapping[label]].values,
    )
    plt.title(f"{label}\nr={r:.2f}, p={p:.3f}")
    plt.xlabel("ML performance")
    plt.ylabel(score_col_mapping[label])
    sns.scatterplot(
        data=df.query("label == @label"),
        x="value",
        y=score_col_mapping[label],
        color=color_mapping[label],
        s=5,
    )
    # plot also the regression line
    sns.regplot(
        data=df.query("label == @label"),
        x="value",
        y=score_col_mapping[label],
        scatter=False,
        color=color_mapping[label],
        
    )
    sns.despine()
    plt.xlabel("ML performance")
    # make x and y ticks smaller
    plt.xticks(fontsize=5)
    plt.yticks(fontsize=5)
    # the tick length should be smaller
    plt.tick_params(axis="both", which="major", length=2)   
    # make axis lines thinner
    plt.gca().spines["bottom"].set_linewidth(0.5)
    plt.gca().spines["left"].set_linewidth(0.5)
plt.savefig(
    os.path.join(FIGURE_PATH, "figure_55_corr_per_clinicalscores.pdf"),
)
