import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from py_neuromodulation import nm_stats


PATH_FIGURES = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf/figures_paper/beta_paper'
PATH_READ = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_std_10s_window_length"
PATH_PER = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/paper_per'

df_all = pd.read_csv(os.path.join(PATH_READ, "all_merged_preprocessed_with_condition_pkgnormed.csv"), index_col=0)
df_all = df_all[df_all["condition"] == "stim_off"]
df_all = df_all.drop(columns=["condition"])
df_all["pkg_dt"] = pd.to_datetime(df_all["pkg_dt"], utc=True).dt.tz_convert("US/Pacific")
df_all["h"] = df_all["pkg_dt"].dt.hour
# restrict only to 8 to 20 hours
df_all = df_all.query("h >= 8 and h <= 20")

list(df_all.columns)

with open('/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/paper_per/LOHO_main_pkg_bk_CLASS_False_loc_ecog_stn_nonorm_withpsd.pkl', "rb") as f:
    d_out = pickle.load(f)

ind_peaks_long = {
    "rcs02l" : 20,
    "rcs02r" : 18,
    "rcs03l" : 13.5,
    "rcs05l" : 25,
    "rcs05r" : 25,
    "rcs06l" : 20,
    "rcs06r" : 18,
    "rcs07l" : 20,
    "rcs07r" : 20,
    "rcs08l" : 20,
    "rcs08r" : 20,
    "rcs09l" : 23,
    "rcs09r" : 23,
    "rcs10l" : 27,
    "rcs10r" : 30,
    "rcs11l" : 25,
    "rcs11r" : 17,
    "rcs12l" : 27,
    "rcs12r" : 20,
    "rcs14l" : 25,
    "rcs15l" : 18,
    "rcs15r" : 18,
    "rcs17l" : 25,
    "rcs17r" : 27,
    "rcs18l" : 23,
    "rcs18r" : 23,
    "rcs19l" : 22,
    "rcs19r" : 25,
    "rcs20l" : 18,
    "rcs20r" : 17,
}

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

df_per_ml = pd.read_csv('/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/paper_per/abc/df_main.csv')
df_per_ml = df_per_ml.drop("Unnamed: 0", axis=1)
df_per_ml = pd.read_csv(os.path.join(PATH_PER, "df_n.csv"))
df_per_ml = df_per_ml.query("include_night == False and CLASSIFICATION == False")

def corr_func(x, y, spearman=True):
    if spearman:
        return stats.spearmanr(x, y).correlation
    else:
        return np.corrcoef(x, y)[0, 1]

SPEARMAN = False
l_per = []
for sub in ind_peaks_short.keys():
    power_sum = df_all[df_all["sub"] == sub][[f"ch_subcortex_welch_psd_{int(i)}_mean" for i in range(1, 115)]].apply(lambda x: 10**x).sum(axis=1).values
    
    ind_beta = df_all[df_all["sub"] == sub][f"ch_subcortex_welch_psd_{ind_peaks_short[sub]}_mean"].apply(lambda x: 10**x).values
    ind_beta_long = df_all[df_all["sub"] == sub][f"ch_subcortex_welch_psd_{int(ind_peaks_long[sub])}_mean"].apply(lambda x: 10**x).values
    low_beta = df_all[df_all["sub"] == sub]["ch_subcortex_fft_low beta_mean_mean"].apply(lambda x: 10**x).values
    high_beta = df_all[df_all["sub"] == sub]["ch_subcortex_fft_high beta_mean_mean"].apply(lambda x: 10**x).values
    all_beta = df_all[df_all["sub"] == sub][[f"ch_subcortex_welch_psd_{i}_mean" for i in range(8, 31)]].apply(
        lambda x: 10**x
    ).mean(axis=1).values
    pkg_bk = df_all[df_all["sub"] == sub]["pkg_bk"].values

    msk_ = ~np.isnan(pkg_bk)
    ind_beta = ind_beta[msk_]
    power_sum = power_sum[msk_]
    ind_beta_long = ind_beta_long[msk_]
    pkg_bk = pkg_bk[msk_]
    low_beta = low_beta[msk_]
    high_beta = high_beta[msk_]
    all_beta = all_beta[msk_]
    per_low_beta = corr_func(low_beta / power_sum, pkg_bk, spearman=SPEARMAN)
    per_high_beta = corr_func(high_beta / power_sum, pkg_bk, spearman=SPEARMAN)
    per_all_beta = corr_func(all_beta / power_sum, pkg_bk, spearman=SPEARMAN)
    per_ind_beta = corr_func(ind_beta / power_sum, pkg_bk, spearman=SPEARMAN)
    per_ind_beta_long = corr_func(ind_beta_long / power_sum, pkg_bk, spearman=SPEARMAN)
    per_ml = df_per_ml.query(f"sub == '{sub}' and pkg_label == 'pkg_bk'")["per"].values[0]

    l_per.append({
        "sub": sub,
        "per_low_beta": per_low_beta,
        "per_high_beta": per_high_beta,
        "per_all_beta": per_all_beta,
        "per_ind_beta": per_ind_beta,
        "per_ind_beta_long": per_ind_beta_long,
        "per_ml": per_ml,
        #"peak_freq": ind_peaks_short[sub]
    })
    
df_per = pd.DataFrame(l_per)

df_per["has_peak"] = True
# if sub in rcs05r, rcs07r, rcs12r, rcs18l, rcs20l then has_peak is False
df_per.loc[df_per["sub"].isin(["rcs05r", "rcs07r", "rcs12r", "rcs18l", "rcs20l"]), "has_peak"] = False

df_per["bg_loc"] = "STN"
subjects_GP = ["09l", "09r", "10l", "10r", "14l", "19l", "19r"]
# if sub in subjects_GP then bg_loc is GP
df_per.loc[df_per["sub"].str.contains("|".join(subjects_GP)), "bg_loc"] = "GP"

# pivot table that contains the performance data
df_per = df_per.melt(id_vars=["sub", "has_peak", "bg_loc"], value_vars=["per_low_beta", "per_high_beta", "per_all_beta", "per_ind_beta", "per_ind_beta_long", "per_ml"], var_name="model", value_name="per")

# delete the rows where had_peak is True and model = per_ind_beta
df_per = df_per.query("not (has_peak == False and model == 'per_ind_beta')")

per_group = df_per.groupby("model")["per"].mean()
order_ = ["per_low_beta", "per_high_beta", "per_all_beta", "per_ind_beta", "per_ml"]
df_per.groupby("model")["per"].std()

#panel a: 
#box plots one data point per hemisphere, only hemispheres with distinct beta peaks: 
# Y-axis corr. coeff. ~bradykinesia,
# X-axis ticks:
# "canonical (8-35 Hz alpha/beta", "canonical beta 13-35 Hz", "canonical low beta (13-20 Hz)", "individual peak frequency (~+-5 Hz)"

#panel b: 
#box plots one data point per hemisphere, comparison of hemispheres with vs. without beta peaks:
# Y-axis corr. coeff ~bradykinesia,
# X-axis ticks: "canonical (whatever works best in panel a) - no peaks" vs. "canonical (whatever works best in panel a) - with peaks" vs. "individual peak frequency (~+-5 Hz) - with peaks"

plt.figure(figsize=(12, 6))
plt.subplot(131)
order = ["per_ind_beta", "per_all_beta"]
df_plt_1 = df_per.query("model =='per_ind_beta' or model == 'per_all_beta'")
#df_plt_1 = df_per.query("model != 'per_ml'")
sns.boxplot(y="per", x="model", data=df_plt_1, #  order=order
            showmeans=True, showfliers=False, palette="viridis", boxprops=dict(alpha=0.5))
sns.swarmplot(y="per", x="model", data=df_plt_1, dodge=False, alpha=0.3, legend=True, hue="bg_loc", )  # order=order
plt.ylabel("Spearman's correlation coefficient")
plt.xticks(rotation=45)

plt.subplot(132)
df_plt_2 = df_per.query("bg_loc == 'STN'")
sns.boxplot(y="per", x="model", data=df_plt_2, hue="has_peak", showmeans=True, showfliers=False, palette="viridis", boxprops=dict(alpha=0.5), order=order)
sns.swarmplot(y="per", x="model", data=df_plt_2, hue="has_peak", dodge=True, alpha=0.3, color="black", order=order, legend=False)
plt.ylabel("Spearman's correlation coefficient")
plt.title("STN")

plt.subplot(133)
df_plt_2 = df_per.query("bg_loc == 'GP'")
sns.boxplot(y="per", x="model", data=df_plt_2, hue="has_peak", showmeans=True, showfliers=False, palette="viridis", boxprops=dict(alpha=0.5), order=order)
sns.swarmplot(y="per", x="model", data=df_plt_2, hue="has_peak", dodge=True, alpha=0.3, order=order, legend=False, color="black")
plt.ylabel("Spearman's correlation coefficient")
plt.title("GP")
plt.tight_layout()

plt.savefig(os.path.join(PATH_FIGURES, "figure_beta_peak_comp_fix_0204.pdf"))
plt.show(block=True)

plt.figure(figsize=(3, 5))
df_gs = pd.read_csv(os.path.join(PATH_PER, "grid_search_mean_sub.csv"))
df_1 = df_per.query("model == 'per_all_beta'")[["sub", "per"]]
df_1["type"] = "all beta"
df_2 = df_gs[["sub", "per_ind_band"]]
df_2["type"] = "individual best band"
df_2.rename(columns={"per_ind_band": "per"}, inplace=True)
df_plt3 = pd.concat([df_1, df_2])
order = ["individual best band", "all beta"]
sns.boxplot(y="per", x="type", data=df_plt3, showmeans=True, showfliers=False, palette="viridis", boxprops=dict(alpha=0.5), order=order)
sns.swarmplot(y="per", x="type", data=df_plt3, dodge=True, alpha=0.3, color="black", order=order, legend=False)
plt.ylabel("Correlation coefficient")
# for each subject draw a line between the two points
for sub in df_plt3["sub"].unique():
    per_ind = df_plt3.query("sub == @sub and type == 'individual best band'")["per"].values[0]
    per_all = df_plt3.query("sub == @sub and type == 'all beta'")["per"].values[0]
    plt.plot([0, 1], [per_ind, per_all], color="gray", alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "figure_beta_ind_gs_best.pdf"))

plt.figure(figsize=(3, 5))
plt.subplot(121)
sns.boxplot(y="range_low", data=df_gs, showmeans=True, showfliers=False, palette="viridis", boxprops=dict(alpha=0.5))
sns.swarmplot(y="range_low", data=df_gs, dodge=True, alpha=0.3, color="black", legend=False)
plt.subplot(122)
sns.boxplot(y="range_high", data=df_gs, showmeans=True, showfliers=False, palette="viridis", boxprops=dict(alpha=0.5))
sns.swarmplot(y="range_high", data=df_gs, dodge=True, alpha=0.3, color="black", legend=False)
plt.tight_layout()



per_all_beta = df_plt3.query("type == 'all beta'")["per"].values
per_ind_beta = df_plt3.query("type == 'individual best band'")["per"].values
nm_stats.permutationTest_relative(per_all_beta, per_ind_beta, False, None, 5000)  # 0.8


per_allbeta = df_per.query("model == 'per_all_beta'")["per"].values
per_indpeak = df_per.query("model == 'per_ind_beta'")["per"].values
nm_stats.permutationTest(per_allbeta, per_indpeak, False, None, 5000)  # 0.2

per_allbeta_peaks = df_per.query("model == 'per_all_beta' and has_peak == True and bg_loc=='STN'")["per"].values
per_allbeta_no_peaks = df_per.query("model == 'per_all_beta' and has_peak == False and bg_loc=='STN'")["per"].values
nm_stats.permutationTest(per_allbeta_peaks, per_allbeta_no_peaks, False, None, 5000)  # 0.87

subs_ind = df_per.query("model == 'per_ind_beta'")["sub"]
per_all_beta_peaks = df_per.query("model == 'per_all_beta' and bg_loc == 'STN' and has_peak == True").query("sub in @subs_ind")["per"].values
per_ind_beta_peaks = df_per.query("model == 'per_ind_beta' and bg_loc == 'STN' and has_peak == True").query("sub in @subs_ind")["per"].values
nm_stats.permutationTest_relative(per_all_beta_peaks, per_ind_beta_peaks, False, None, 5000)  # 0.016

per_allbeta_peaks = df_per.query("model == 'per_all_beta' and has_peak == True and bg_loc=='GP'")["per"].values
per_indbeta_peaks = df_per.query("model == 'per_ind_beta' and has_peak == True and bg_loc=='GP'")["per"].values
nm_stats.permutationTest(per_allbeta_peaks, per_indbeta_peaks, False, None, 5000)  # 0.35





plt.figure(figsize=(12, 6))
plt.subplot(131)
order_sorted_best = df_per.query("model != 'per_ml'").groupby("model")["per"].mean().sort_values().index
sns.boxplot(y="per", x="model", data=df_per.query("model != 'per_ml'"), hue="bg_loc",
            showmeans=True, showfliers=False, order=order_sorted_best, palette="viridis", boxprops=dict(alpha=0.5))
sns.swarmplot(y="per", x="model", data=df_per.query("model != 'per_ml'"), hue="bg_loc", dodge=True, alpha=0.3, color="black",
                order=order_sorted_best, legend=False)
plt.ylabel("Correlation coefficient")
plt.xticks(rotation=45)
plt.subplot(132)
df_plt_2 = df_per.query("model == 'per_high_beta' or model == 'per_ind_beta'").query("bg_loc == 'STN'")
order_2 = ["per_ind_beta", "per_high_beta"]
sns.boxplot(y="per", x="model", data=df_plt_2, hue="has_peak", showmeans=True, showfliers=False, palette="viridis", boxprops=dict(alpha=0.5), order=order_2)
sns.swarmplot(y="per", x="model", data=df_plt_2, hue="has_peak", dodge=True, alpha=0.3, color="black", order=order_2, legend=False)
plt.ylabel("Correlation coefficient")
plt.title("STN")

plt.subplot(133)
df_plt_2 = df_per.query("model == 'per_high_beta' or model == 'per_ind_beta'").query("bg_loc == 'GP'")
order_2 = ["per_ind_beta", "per_high_beta"]
sns.boxplot(y="per", x="model", data=df_plt_2, hue="has_peak", showmeans=True, showfliers=False, palette="viridis", boxprops=dict(alpha=0.5), order=order_2)
sns.swarmplot(y="per", x="model", data=df_plt_2, hue="has_peak", dodge=True, alpha=0.3, order=order_2, legend=False, color="black")
plt.ylabel("Correlation coefficient")
plt.title("GP")
plt.savefig(os.path.join(PATH_FIGURES, "figure_beta_peak_comp_fix.pdf"))
plt.show(block=True)


df_per.query("model == 'per_ml'")["per"].values
subs_ind = df_per.query("model == 'per_ind_beta'")["sub"]
per_highbeta = df_per.query("model == 'per_high_beta' and bg_loc == 'STN'").query("sub in @subs_ind")["per"].values
per_indpeak = df_per.query("model == 'per_ind_beta' and bg_loc == 'STN'")["per"].values
nm_stats.permutationTest_relative(per_highbeta, per_indpeak, False, None, 5000)  # 0.024

per_lowbeta = df_per.query("model == 'per_low_beta' and bg_loc == 'STN'").query("sub in @subs_ind")["per"].values
per_indpeak = df_per.query("model == 'per_ind_beta' and bg_loc == 'STN'")["per"].values
nm_stats.permutationTest_relative(per_lowbeta, per_indpeak, False, None, 5000)  # 0.4082

per_highbeta = df_per.query("model == 'per_high_beta' and bg_loc == 'GP'").query("sub in @subs_ind")["per"].values
per_lowbeta = df_per.query("model == 'per_low_beta' and bg_loc == 'GP'").query("sub in @subs_ind")["per"].values
per_indpeak = df_per.query("model == 'per_ind_beta' and bg_loc == 'GP'")["per"].values
nm_stats.permutationTest_relative(per_highbeta, per_indpeak, False, None, 5000)  # 0.3
nm_stats.permutationTest_relative(per_lowbeta, per_indpeak, False, None, 5000)  # 0.85
# there is a significant difference between high beta and individual peak beta in the STN but not GP

per_highbeta_peakTrue= df_per.query("model == 'per_high_beta' and bg_loc == 'STN' and has_peak == True")["per"].values
per_highbeta_peakFalse = df_per.query("model == 'per_high_beta' and bg_loc == 'STN' and has_peak == False")["per"].values
nm_stats.permutationTest(per_highbeta_peakTrue, per_highbeta_peakFalse, False, None, 5000)  # 0.8
# there is no sig. difference between high beta with and without peak, in the GP all had a peak

per_highbeta = df_per.query("model == 'per_high_beta' and bg_loc == 'STN'")
per_lowbeta = df_per.query("model == 'per_low_beta' and bg_loc == 'STN'")
nm_stats.permutationTest_relative(per_highbeta["per"].values, per_lowbeta["per"].values, False, None, 5000)  # 0.22

per_allbeta = df_per.query("model == 'per_all_beta' and bg_loc == 'STN'")
nm_stats.permutationTest_relative(per_highbeta["per"].values, per_allbeta["per"].values, False, None, 5000)  # 0.76

per_highbeta = df_per.query("model == 'per_high_beta' and bg_loc == 'GP'")
per_lowbeta = df_per.query("model == 'per_low_beta' and bg_loc == 'GP'")
nm_stats.permutationTest_relative(per_highbeta["per"].values, per_lowbeta["per"].values, False, None, 5000)  # 0.64

per_allbeta = df_per.query("model == 'per_low_beta' and bg_loc == 'GP'")
nm_stats.permutationTest_relative(per_highbeta["per"].values, per_allbeta["per"].values, False, None, 5000)  # 0.64


nm_stats.permutationTest_relative(df_per.query("model == 'per_ind_beta'")["per"].values, df_per.query("model == 'per_high_beta'")["per"].values, False, None, 5000)


nm_stats.permutationTest_relative(df_per.query("model == 'per_ml'")["per"].values, df_per.query("model == 'per_ind_beta'")["per"].values, False, None, 5000)
nm_stats.permutationTest_relative(df_per.query("model == 'per_ml'")["per"].values, df_per.query("model == 'per_high_beta'")["per"].values, False, None, 5000)
nm_stats.permutationTest_relative(df_per.query("model == 'per_ind_beta'")["per"].values, df_per.query("model == 'per_all_beta'")["per"].values, False, None, 5000)
nm_stats.permutationTest_relative(df_per.query("model == 'per_ind_beta'")["per"].values, df_per.query("model == 'per_ind_beta_long'")["per"].values, False, None, 5000)

plt.figure(figsize=(3/2, 8.2/3))
sns.boxplot(data=df_per, x="model", y="per", palette="viridis", showmeans=True, showfliers=False, boxprops=dict(alpha=0.5), order=order_)
sns.swarmplot(data=df_per, x="model", y="per", dodge=False, palette="viridis", alpha=0.3, s=2, order=order_)
plt.ylabel("Correlation coefficient")
plt.title("Beta prediction")
#plt.savefig(os.path.join(PATH_FIGURES, "figure_beta_comp.pdf"))
plt.show(block=True)

df_ind_peaks = pd.DataFrame(ind_peaks_short.items(), columns=["sub", "peak_freq"])
df_ind_peaks["peak_freq"].mean()
df_ind_peaks["peak_freq"].std()
plt.figure(figsize=(3, 8.2/3))
sns.boxplot(data=df_ind_peaks, y="peak_freq", palette="viridis", showmeans=True, showfliers=False, boxprops=dict(alpha=0.5)
)
sns.swarmplot(data=df_ind_peaks, y="peak_freq", dodge=False, palette="viridis", alpha=0.9, s=5)
plt.ylabel("Frequency [Hz]")
plt.title("Individual peak frequency")
plt.savefig(os.path.join(PATH_FIGURES, "figure_ind_peaks.pdf"))
plt.show(block=True)

# Plot example prediction beta vs decoding

df_ = pd.DataFrame(l_per)
df_["diff"] = df_["per_ml"] - df_["per_ind_beta"]

sub = "rcs02r"
df_sub = df_all[df_all["sub"] == sub]
ind_beta = df_sub[f"ch_subcortex_welch_psd_{ind_peaks_short[sub]}_mean"].values
pkg_bk = df_sub["pkg_bk"].values
# get non-nan values
mask = ~np.isnan(pkg_bk)
ind_beta = ind_beta[mask]
pkg_bk = pkg_bk[mask]
# z-score normalize each
ind_beta = stats.zscore(ind_beta).clip(-2, 2.5)
pkg_bk = stats.zscore(pkg_bk).clip(-2, 2.5)

window_size = 5
weights = np.ones(window_size) / window_size  # Equal weights

corr_ = np.corrcoef(ind_beta, pkg_bk)[0, 1]
ml_pred = d_out[sub]["pr"]
ml_pred = stats.zscore(ml_pred).clip(-2, 2.5)

plt.figure(figsize=(10, 4))
plt.subplot(211)
corr_ml = np.corrcoef(ml_pred, pkg_bk)[0, 1]
plt.plot(pkg_bk, label="Bradykinesia", color="black",)
plt.plot(ind_beta*-1 , label="Low beta", color="red", alpha=0.6)
plt.legend()

plt.subplot(212)
plt.plot(pkg_bk, label="Bradykinesia", color="black",)
plt.plot(ml_pred, label="ML prediction", color="blue", alpha=0.6)
plt.legend()
plt.ylim(-2.7, 2.7)
plt.suptitle(f"Subject: {sub} - Correlation: {corr_:.2f} - ML: {corr_ml:.2f}")
plt.savefig(os.path.join(PATH_FIGURES, "beta_prediction_example.pdf"))
plt.show(block=True)

subs = df_all["sub"].unique()

performance_data = []

pdf_path = "beta_prediction_plots.pdf"
with PdfPages(pdf_path) as pdf:
    for sub in subs:
        df_sub = df_all[df_all["sub"] == sub]
        
        low_beta = df_sub["ch_subcortex_fft_low beta_mean_mean"]
        high_beta = df_sub["ch_subcortex_fft_high beta_mean_mean"]
        pkg_dk = df_sub["pkg_dk"]
        pkg_bk = df_sub["pkg_bk"]
        pkg_tremor = df_sub["pkg_tremor"]

        # z-score normalize each
        low_beta = stats.zscore(low_beta.values)
        high_beta = stats.zscore(high_beta.values)
        pkg_dk = stats.zscore(pkg_dk.values)
        pkg_bk = stats.zscore(pkg_bk.values)
        pkg_tremor = stats.zscore(pkg_tremor.values)

        plt_cnt = 1
        plt.figure(figsize=(10, 10))
        for pkg, label in zip([pkg_dk, pkg_bk, pkg_tremor], ["Dyskinesia", "Bradykinesia", "Tremor"]):
            plt.subplot(3, 2, plt_cnt)
            plt.plot(low_beta, label="Low beta")
            plt.plot(pkg, label=label)
            corr_ = np.corrcoef(low_beta, pkg)[0, 1]
            plt.title(f"Low beta vs {label} - corr: {corr_:.2f}")
            plt.legend()
            plt_cnt += 1
            plt.subplot(3, 2, plt_cnt)
            plt.plot(high_beta, label="High beta")
            plt.plot(pkg, label=label)
            corr_ = np.corrcoef(high_beta, pkg)[0, 1]
            plt.title(f"High beta vs {label} - corr: {corr_:.2f}")
            plt_cnt += 1
            plt.legend()
            plt.tight_layout()
            plt.suptitle(f"Subject: {sub}")

            # Save performance data
            performance_data.append({
                "subject": sub,
                "condition": label,
                "low_beta_corr": np.corrcoef(low_beta, pkg)[0, 1],
                "high_beta_corr": np.corrcoef(high_beta, pkg)[0, 1]
            })

        # Save the current figure to the PDF
        pdf.savefig()
        plt.close()

# Convert performance data to DataFrame
performance_df = pd.DataFrame(performance_data)
performance_df.to_csv("beta_prediction_performance.csv", index=False)

# make a scatter plot of the performance data
plt.figure(figsize=(10, 7))
plt.subplot(121)
sns.boxplot(data=performance_df, x="condition", y="low_beta_corr", palette="viridis", showmeans=True, showfliers=False)
sns.swarmplot(data=performance_df, x="condition", y="low_beta_corr", dodge=False, palette="viridis", alpha=0.9, s=2)
plt.ylabel("Correlation coefficient")
plt.title("Low beta")
plt.subplot(122)
sns.boxplot(data=performance_df, x="condition", y="high_beta_corr", palette="viridis", showmeans=True, showfliers=False)
sns.swarmplot(data=performance_df, x="condition", y="high_beta_corr", dodge=False, palette="viridis", alpha=0.9, s=2)
plt.ylabel("Correlation coefficient")
plt.title("High beta")
plt.tight_layout()
plt.show(block=True)


data = []
for CLASSIFICATION in d_out.keys():
    if CLASSIFICATION:
        per_ = "ba"
    else:
        per_ = "corr_coeff"
    for pkg_decode_label in d_out[CLASSIFICATION].keys():
        for loc in d_out[CLASSIFICATION][pkg_decode_label].keys():
            for sub in d_out[CLASSIFICATION][pkg_decode_label][loc].keys():
                data.append({
                    "CLASSIFICATION": CLASSIFICATION,
                    "per": d_out[CLASSIFICATION][pkg_decode_label][loc][sub][per_],
                    "sub": sub,
                    "pkg_decode_label": pkg_decode_label,
                    "loc": loc
                })

df = pd.DataFrame(data)
res_df = df[df["CLASSIFICATION"] == False].query("loc == 'ecog_stn'")

plt.figure(figsize=(10, 7))
sns.boxplot(data=res_df, x="pkg_decode_label", y="per", palette="viridis", showmeans=True, showfliers=False)
sns.swarmplot(data=res_df, x="pkg_decode_label", y="per", dodge=False, palette="viridis", alpha=0.9, s=2)
plt.ylabel("Correlation coefficient")
plt.title("STN")
plt.show(block=True)

res_df["model"] = "ML"
# rename condition to pkg_decode_label
performance_df["pkg_decode_label"] = performance_df["condition"].replace({"Dyskinesia": "pkg_dk", "Bradykinesia": "pkg_bk", "Tremor": "pkg_tremor"}) 
# remove condition column
performance_df = performance_df.drop("condition", axis=1)
# remame subject to sub
performance_df = performance_df.rename(columns={"subject": "sub"})
# rename low_beta_corr to per
performance_df = performance_df.rename(columns={"low_beta_corr": "per"})
performance_df["model"] = "Low beta prediction"
performance_df["loc"] = "stn"

# concatenate the two dataframes
df_concat = pd.concat([res_df, performance_df], axis=0)
# take abs value of per
df_concat["per"] = df_concat["per"].abs()

plt.figure(figsize=(10, 7), dpi=300)
sns.boxplot(data=df_concat, x="pkg_decode_label", y="per", hue="model", palette="viridis", showmeans=True, showfliers=False, boxprops=dict(alpha=0.5), order=["pkg_bk", "pkg_dk", "pkg_tremor"])
sns.swarmplot(data=df_concat, x="pkg_decode_label", y="per", hue="model", dodge=True, palette="viridis", alpha=0.9, s=5, order=["pkg_bk", "pkg_dk", "pkg_tremor"])
plt.ylabel("Correlation coefficient")
plt.title("STN")
plt.savefig(os.path.join(PATH_FIGURES, "figure_37_beta_comparison_to_ML.pdf"))
plt.show(block=True)

# take the difference between the two models for each subject and condition
df_diff = df_concat.pivot_table(index=["sub", "pkg_decode_label"], columns="model", values="per").reset_index()
df_diff["diff"] = df_diff["Low beta prediction"] - df_diff["ML"]
df_diff = df_diff.dropna()


plt.figure(figsize=(5, 7))
sns.boxplot(data=df_diff, x="pkg_decode_label", y="diff", palette="viridis", showmeans=True, showfliers=False, boxprops=dict(alpha=0.5))
sns.swarmplot(data=df_diff, x="pkg_decode_label", y="diff", dodge=False, palette="viridis", alpha=0.9, s=5)
plt.ylabel("Difference in correlation coefficient")
mean_diff = df_diff.groupby(["pkg_decode_label"])["diff"].mean()
std_diff = df_diff.groupby(["pkg_decode_label"])["diff"].std()
str_title = f"Performance differences\nBK: {mean_diff['pkg_bk'].round(2)} ± {std_diff['pkg_bk'].round(2)}\nDK: {mean_diff['pkg_dk'].round(2)} ± {std_diff['pkg_dk'].round(2)}\nTremor: {mean_diff['pkg_tremor'].round(2)} ± {std_diff['pkg_tremor'].round(2)}"

perc_bk = np.sum(df_diff.query("pkg_decode_label == 'pkg_bk'")["diff"] < 0) / len(df_diff.query("pkg_decode_label == 'pkg_bk'"))
perc_dk = np.sum(df_diff.query("pkg_decode_label == 'pkg_dk'")["diff"] < 0) / len(df_diff.query("pkg_decode_label == 'pkg_dk'"))
perc_tremor = np.sum(df_diff.query("pkg_decode_label == 'pkg_tremor'")["diff"] < 0) / len(df_diff.query("pkg_decode_label == 'pkg_tremor'"))
str_title += f"\nPercentage ML better BK: {perc_bk:.2f} DK: {perc_dk:.2f} Tremor: {perc_tremor:.2f}"

plt.title(str_title)
plt.show(block=True)

