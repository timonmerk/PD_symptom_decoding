import pickle
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os
from py_neuromodulation import nm_stats

PATH_FIGURES = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf/figures_paper'

ind_peaks_long = {
    "rcs02l" : 20,              # STN
    "rcs02r" : 18,              # STN
    "rcs03l" : 13.5,            # STN
    "rcs05l" : 26,              # STN
    "rcs05r" : 26,              # STN
    "rcs06l" : 28,              # STN
    "rcs06r" : 18,              # STN
    "rcs07l" : 13,              # STN
    "rcs07r" : 8,               # STN
    "rcs08l" : 25,  # none      # STN
    "rcs08r" : 27,              # STN
    "rcs09l" : 24,              # GP
    "rcs09r" : 23,              # GP
    "rcs10l" : 27, # none       # GP
    "rcs10r" : 29,              # GP
    "rcs11l" : 27,              # STN
    "rcs11r" : 25,              # STN
    "rcs12l" : 28,              # STN
    "rcs12r" : 28, # none       # STN
    "rcs14l" : 25,              # STN
    "rcs15l" : 22,              # STN
    "rcs15r" : 18,              # STN
    "rcs17l" : 27,              # STN
    "rcs17r" : 29,              # STN
    "rcs18l" : 23, # none       # STN
    "rcs18r" : 23,              # STN
    "rcs19l" : 10,              # GP
    "rcs19r" : 22,              # GP
    "rcs20l" : 17,              # STN
    "rcs20r" : 17,              # STN
}

subs_GP = ["rcs09l", "rcs09r", "rcs10l", "rcs10r", "rcs14r", "rcs19l", "rcs19r"]

PATH_PER = "publication_figures"
LOAD_ECOG = False

if LOAD_ECOG is False:
    df_comp_withnight = pd.read_csv(os.path.join(PATH_PER, "df_comp_beta_ml_incl_night.csv"))
else:
    #df_comp_withnight = pd.read_csv(os.path.join(PATH_PER, "df_comp_beta_ml_incl_night_ecog.csv"))
    print("")

if LOAD_ECOG is False:
    df_comp = pd.read_csv(os.path.join(PATH_PER, "df_comp_beta_ml.csv"))
else:
    df_comp = pd.read_csv(os.path.join(PATH_PER, "df_comp_beta_ml_ecog.csv"))

df_comp["loc"] = "STN"
df_comp.loc[df_comp["sub"].isin(subs_GP), "loc"] = "GP"


plt.figure(figsize=(5, 3))

for idx, label_ in enumerate(["pkg_bk", "pkg_tremor", "pkg_dk"]): 
    plt.subplot(1, 3, idx + 1)
    if label_ == "pkg_dk":
        df_sc_non_working = df_comp.query("label == @label_ and type == 'corr_ind' and (peak_present == 0 or value > 0)")
        df_sc_non_working["value"] *= -1  # invert values for plotting
    else:
        df_sc_non_working = df_comp.query("label == @label_ and type == 'corr_ind' and (peak_present == 0 or value < 0)")

    df_subs_sc_non_working = df_sc_non_working["sub"].unique()
    df_sc_working = df_comp.query("label == @label_ and type == 'corr_pr' and sub in @df_subs_sc_non_working")

    df_plt = pd.concat([df_sc_non_working, df_sc_working])

    sns.boxplot(x="type", y="value", data=df_plt, showfliers=False, showmeans=True, boxprops={"alpha": 0.5})
    sns.swarmplot(x="type", y="value", data=df_plt, dodge=False, color=".25", alpha=0.5, hue="peak_present", palette=["red", "gray"], legend=False)

    # Draw connection lines per subject (assuming 'subject' column exists)
    for subject, sub_df in df_plt.groupby("sub"):
        sub_df_sorted = sub_df.sort_values("type")
        if sub_df_sorted["value"].notna().all() and len(sub_df_sorted) > 1:
            x_vals = [list(df_plt["type"].unique()).index(t) for t in sub_df_sorted["type"]]
            y_vals = sub_df_sorted["value"].values
            plt.plot(x_vals, y_vals, color="gray", alpha=0.4, linewidth=1)

    plt.title(f"{label_}")
    
    plt.xticks([0, 1], ["Beta", "Decoder"])
    plt.ylim(-0.5, 0.9)
    if idx != 0:
        plt.yticks([])
    else:
        plt.ylabel("Pearson correlation coefficient [r]")
    # turn off upper and right spines
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, f"figure_54_comp_nonsigbeta_patients_all.pdf"))
plt.show()

