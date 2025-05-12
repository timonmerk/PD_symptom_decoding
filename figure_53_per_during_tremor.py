import pickle
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os

ind_peaks_short = {
    "rcs02l" : 20,
    "rcs02r" : 18,
    "rcs03l" : 27,
    "rcs05l" : 27,
    "rcs05r" : 25,  # NONE
    "rcs06l" : 30,
    "rcs06r" : 18,
    "rcs07l" : 27,
    "rcs07r" : 20, # None
    "rcs08l" : 15,
    "rcs08r" : 25,
    "rcs09l" : 22,
    "rcs09r" : 22,
    "rcs10l" : 27,
    "rcs10r" : 27,
    "rcs11l" : 27, # DOUBLE PEAK
    "rcs11r" : 17,
    "rcs12l" : 27,
    "rcs12r" : 20, # NONE
    "rcs14l" : 25, 
    "rcs15l" : 18,
    "rcs15r" : 18,
    "rcs17l" : 30,
    "rcs17r" : 30,
    "rcs18l" : 23, # NONE
    "rcs18r" : 23,
    "rcs19l" : 22,
    "rcs19r" : 25,
    "rcs20l" : 18, # NONE
    "rcs20r" : 17,
}


PATH_FIGURES = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf/figures_paper'
PATH_PER = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/paper_per"

PATH_PER = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/paper_per"

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

df_comp = []
for label in ["pkg_dk", "pkg_bk", "pkg_tremor"]:

    df_features = df_features_orig.copy()
    mask = ~df_features[label].isnull()
    df_features = df_features[mask].copy()
    df_features = df_features.drop(columns=df_features.columns[df_features.isnull().all()])
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.dropna(axis=1)

    file = f"LOHO_main_{label}_CLASS_False_loc_ecog_stn_nonorm_withpsd.pkl"
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

        def compute_corrs(df_sub, pr_, true_, DURING_TREMOR: bool):
            df_sub_ = df_sub.copy()
            if DURING_TREMOR:
                df_sub_ = df_sub_[df_sub_["pkg_tremor_class"] == True]
                pr__ = stats.zscore(pr_[df_sub["pkg_tremor_class"] == True])
                true__ = stats.zscore(true_[df_sub["pkg_tremor_class"] == True])
            else:
                df_sub_ = df_sub_[df_sub_["pkg_tremor_class"] == False]
                pr__ = stats.zscore(pr_[df_sub["pkg_tremor_class"] == False])
                true__ = stats.zscore(true_[df_sub["pkg_tremor_class"] == False])
            ind_band = df_sub_[[f"ch_subcortex_welch_psd_{int(i)}_mean" for i in range(int(ind_peaks_short[sub]-2.5), int(ind_peaks_short[sub]+2.5))]].apply(lambda x: 10**x).mean(axis=1).values
            power_sum = df_sub_[[f"ch_subcortex_welch_psd_{int(i)}_mean" for i in range(1, 115)]].apply(lambda x: 10**x).sum(axis=1).values

            ind_band = stats.zscore(ind_band)

            if CORR_SPEARMANS is False:
                corr_ind = np.corrcoef(ind_band / power_sum, true__)[0, 1]
                corr_pr = np.corrcoef(pr__, true__)[0, 1]
            else:
                corr_pr = stats.spearmanr(pr_, true__).correlation
                corr_ind = stats.spearmanr(ind_band  / power_sum, true__).correlation

            return corr_ind, corr_pr

        corr_ind, corr_pr = compute_corrs(df_sub, pr_, true_, False)
        df_comp.append({
            "sub" : sub,
            "corr_ind" : corr_ind,
            "corr_pr" : corr_pr,
            "label": label,
            "during_tremor": False
        })

        corr_ind, corr_pr = compute_corrs(df_sub, pr_, true_, True)
        df_comp.append({
            "sub" : sub,
            "corr_ind" : corr_ind,
            "corr_pr" : corr_pr,
            "label": label,
            "during_tremor": True
        })

    if PLT_:
        pdf_pages.close()

df_comp = pd.DataFrame(df_comp)
# create single column "per", and another column indicating if it's either corr_ind or corr_pr
df_comp = df_comp.melt(id_vars=["sub", "label", "during_tremor"], value_vars=["corr_ind", "corr_pr"], var_name="type", value_name="value")
order_ = ["pkg_bk", "pkg_tremor", "pkg_dk"]


plt.figure(figsize=(5, 5))

hue_order = [True, False]
df_plt_ = df_comp.copy()
plt.subplot(121)
sns.boxplot(data=df_plt_.query("type == 'corr_pr'"), x="label", y="value",
            hue="during_tremor", palette="viridis", boxprops=dict(alpha=.3),
            showfliers=False, showmeans=True, order=order_, hue_order=hue_order)
sns.swarmplot(data=df_plt_.query("type == 'corr_pr'"), x="label", y="value",
              hue="during_tremor", dodge=True, color="black", alpha=.5,
              order=order_, palette="viridis", legend=False, hue_order=hue_order)
plt.title("LOHO - Tremor Analysis - ML")
if CORR_SPEARMANS:
    plt.ylabel("Spearman's correlation")
else:
    plt.ylabel("Pearson's correlation")

plt.subplot(122)
sns.boxplot(data=df_plt_.query("type == 'corr_ind'"), x="label", y="value",
            hue="during_tremor", palette="viridis", boxprops=dict(alpha=.3),
            showfliers=False, showmeans=True, order=order_, hue_order=hue_order)
sns.swarmplot(data=df_plt_.query("type == 'corr_ind'"), x="label", y="value",
              hue="during_tremor", dodge=True, color="black", alpha=.5,
              order=order_, palette="viridis", legend=False, hue_order=hue_order)
plt.title("LOHO - Tremor Analysis - Individual Peaks")
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