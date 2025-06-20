import pickle
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os

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


PATH_FIGURES = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf/figures_paper/figures_final'
PATH_PER = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/paper_per/without_night"

#df_features = pd.read_csv("/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_std_10s_window_length/all_merged_preprocessed_with_condition.csv")
df_features = pd.read_csv(os.path.join("/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_std_10s_window_length", "all_merged_preprocessed_with_condition_pkgnormed.csv"), index_col=0)

df_features = df_features[df_features["condition"] == "stim_off"]
df_features["pkg_dt"] = pd.to_datetime(df_features["pkg_dt"], utc=True).dt.tz_convert("US/Pacific")
df_features["h"] = df_features["pkg_dt"].dt.hour

df_features_orig = df_features.copy()

df_features_orig = df_features_orig.query("h >= 8 and h <= 20")

PLT_ = False
CORR_SPEARMANS = False
if PLT_:
    pdf_pages = PdfPages(os.path.join(PATH_FIGURES, f"predictions_beta_ml.pdf"))

label_class = "pkg_tremor_class"
label_class = "pkg_dk_class"
#label_class = "pkg_tremor_class"
df_comp = []
for label in ["pkg_bk", "pkg_tremor", "pkg_dk"]:

    df_features = df_features_orig.copy()
    mask = ~df_features[label].isnull()
    df_features = df_features[mask].copy()
    df_features = df_features.drop(columns=df_features.columns[df_features.isnull().all()])
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.dropna(axis=1)

    file = f"LOHO_main_{label}_CLASS_False_loc_ecog_stn_withpsd.pkl"
    with open(f"{PATH_PER}/{file}", "rb") as f:
        d_out = pickle.load(f)

    for sub in np.sort(list(d_out.keys())):

        pr_ = d_out[sub]["pr"]
        true_ = d_out[sub]["y_"]
        time_pr = d_out[sub]["time"]
        df_sub = df_features.query(f"sub == '{sub}'")
        time_band = pd.to_datetime(df_sub["pkg_dt"], utc=True).dt.tz_convert("US/Pacific")
        common_time = np.intersect1d(time_pr, time_band.values)
        pr_ = pr_[np.isin(time_pr, common_time)]
        true_ = true_[np.isin(time_pr, common_time)]
        time_band = time_band[np.isin(time_band.values, common_time)]
        time_idx = time_band.index
        df_sub = df_sub.loc[time_idx]
        df_sub["pkg_tremor_class"] = df_sub["pkg_tremor"] > 1
        # y = (df_sub[label].copy() / df_sub[label].max()) > 0.02
        df_sub["pkg_dk_class"] = (df_sub["pkg_dk"] / df_sub["pkg_dk"].max() > 0.02)
        df_sub["pkg_bk_class"] = df_sub["pkg_bk"] > 50

        def compute_corrs(df_sub, pr_, true_, label_ = "pkg_tremor_class", DURING_TREMOR: bool=True):
            df_sub_ = df_sub.copy()
            if DURING_TREMOR:
                df_sub_ = df_sub_[df_sub_[label_] == True]
                pr__ = pr_[df_sub[label_] == True]
                true__ = true_[df_sub[label_] == True]
            else:
                df_sub_ = df_sub_[df_sub_[label_] == False]
                pr__ = pr_[df_sub[label_] == False]
                true__ = true_[df_sub[label_] == False]
            ind_band = df_sub_[[f"ch_subcortex_welch_psd_{int(i)}_mean" for i in range(int(ind_peaks_long[sub]-2.5), int(ind_peaks_long[sub]+2.5))]].apply(lambda x: 10**x).mean(axis=1).values
            power_sum = df_sub_[[f"ch_subcortex_welch_psd_{int(i)}_mean" for i in range(1, 115)]].apply(lambda x: 10**x).sum(axis=1).values

            #ind_band = stats.zscore(ind_band)

            if CORR_SPEARMANS is False:
                corr_ind = np.corrcoef(ind_band / power_sum, true__)[0, 1]
                corr_pr = np.corrcoef(pr__, true__)[0, 1]
            else:
                corr_ind = stats.spearmanr(ind_band  / power_sum, true__).correlation
                corr_pr = stats.spearmanr(pr_, true__).correlation

            return corr_ind, corr_pr

        corr_ind, corr_pr = compute_corrs(df_sub, pr_, true_, label_class, False)
        df_comp.append({
            "sub" : sub,
            "corr_ind" : corr_ind,
            "corr_pr" : corr_pr,
            "label": label,
            "during_label_class": False
        })

        corr_ind, corr_pr = compute_corrs(df_sub, pr_, true_, label_class, True)
        df_comp.append({
            "sub" : sub,
            "corr_ind" : corr_ind,
            "corr_pr" : corr_pr,
            "label": label,
            "during_label_class": True
        })

    if PLT_:
        pdf_pages.close()

df_comp = pd.DataFrame(df_comp)
# create single column "per", and another column indicating if it's either corr_ind or corr_pr
df_comp = df_comp.melt(id_vars=["sub", "label", "during_label_class"], value_vars=["corr_ind", "corr_pr"], var_name="type", value_name="value")
order_ = ["pkg_bk", "pkg_tremor", "pkg_dk"]

from py_neuromodulation import nm_stats
nm_stats.permutationTest_relative(
    df_comp.query("label == 'pkg_tremor' and during_label_class == False and type == 'corr_pr'")["value"].values,
    df_comp.query("label == 'pkg_tremor' and during_label_class == True and type == 'corr_pr'")["value"].values,
    False, None, 5000
)  # <10^-5

during_ = df_comp.query("label == 'pkg_tremor' and during_label_class == True and type == 'corr_pr'")["value"].values

nm_stats.permutationTest_relative(
    df_comp.query("label == 'pkg_tremor' and during_label_class == False and type == 'corr_pr'")["value"].values,
    df_comp.query("label == 'pkg_tremor' and during_label_class == True and type == 'corr_pr'")["value"].values,
    False, None, 5000
)  # <10^-5

nm_stats.permutationTest_relative(
    df_comp.query("label == 'pkg_tremor' and during_label_class == False and type == 'corr_pr' and sub != 'rcs05l'")["value"].values,
    df_comp.query("label == 'pkg_tremor' and during_label_class == True and type == 'corr_pr' and sub != 'rcs05l'")["value"].values,
    False, None, 5000
)  # <10^-5


plt.figure(figsize=(5, 5))

hue_order = [True, False]
df_plt_ = df_comp.copy()
plt.subplot(121)
sns.boxplot(data=df_plt_.query("type == 'corr_pr'"), x="label", y="value",
            hue="during_label_class", palette="viridis", boxprops=dict(alpha=.3),
            showfliers=False, showmeans=True, order=order_, hue_order=hue_order)
sns.swarmplot(data=df_plt_.query("type == 'corr_pr'"), x="label", y="value",
              hue="during_label_class", dodge=True, color="black", alpha=.5,
              order=order_, palette="viridis", legend=False, hue_order=hue_order)
plt.title("LOHO - Tremor Analysis - ML")
if CORR_SPEARMANS:
    plt.ylabel("Spearman's correlation")
else:
    plt.ylabel("Pearson's correlation")

plt.subplot(122)
sns.boxplot(data=df_plt_.query("type == 'corr_ind'"), x="label", y="value",
            hue="during_label_class", palette="viridis", boxprops=dict(alpha=.3),
            showfliers=False, showmeans=True, order=order_, hue_order=hue_order)
sns.swarmplot(data=df_plt_.query("type == 'corr_ind'"), x="label", y="value",
              hue="during_label_class", dodge=True, color="black", alpha=.5,
              order=order_, palette="viridis", legend=False, hue_order=hue_order)
plt.title("During DK LOHO - Tremor Analysis - Individual Peaks")
if CORR_SPEARMANS:
    plt.ylabel("Spearman's correlation")
else:
    plt.ylabel("Pearson's correlation")
plt.tight_layout()
#plt.savefig(os.path.join(PATH_FIGURES, "predictions_beta_ml_fig2_pearson_1205.pdf"))
plt.show(block=True)


plt.figure()
df_features["Tremor present"] = df_features["pkg_tremor"] > 1
sns.boxplot(data=df_features, y="pkg_bk", x="Tremor present", palette="viridis", boxprops=dict(alpha=.3),
            showfliers=False, showmeans=True)

sns.lmplot(data=df_features.query("pkg_tremor > 1"), x="pkg_bk", y="pkg_tremor", )# hue="sub"



np.sum((df_comp.query("label == 'pkg_bk' and type == 'corr_pr'")["value"].values - np.abs(df_comp.query("label == 'pkg_bk' and type == 'corr_ind'")["value"].values)) > 0)
np.sum((df_comp.query("label == 'pkg_dk' and type == 'corr_pr'")["value"].values - np.abs(df_comp.query("label == 'pkg_dk' and type == 'corr_ind'")["value"].values)) > 0)
np.sum((df_comp.query("label == 'pkg_tremor' and type == 'corr_pr'")["value"].values - np.abs(df_comp.query("label == 'pkg_tremor' and type == 'corr_ind'")["value"].values)) > 0)

from py_neuromodulation import nm_stats
nm_stats.permutationTest_relative(
    df_plt_.query("label == 'pkg_bk' and type == 'corr_pr'")["value"].values,
    np.abs(df_plt_.query("label == 'pkg_bk' and type == 'corr_ind'")["value"].values),
    False, None, 5000
)  # <10^-5
nm_stats.permutationTest_relative(
    df_plt_.query("label == 'pkg_dk' and type == 'corr_pr'")["value"].values,
    np.abs(df_plt_.query("label == 'pkg_dk' and type == 'corr_ind'")["value"].values),
    False, None, 5000
)  # 0.0004
nm_stats.permutationTest_relative(
    df_plt_.query("label == 'pkg_tremor' and type == 'corr_pr'")["value"].values,
    np.abs(df_plt_.query("label == 'pkg_tremor' and type == 'corr_ind'")["value"].values),
    False, None, 5000
)  # 0.0014