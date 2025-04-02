import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from tqdm import tqdm
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

def corr_func(x, y, spearman=True):
    if spearman:
        return stats.spearmanr(x, y).correlation
    else:
        return np.corrcoef(x, y)[0, 1]

SPEARMAN = True
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

    l_per.append({
        "sub": sub,
        "per_low_beta": per_low_beta,
        "per_high_beta": per_high_beta,
        "per_all_beta": per_all_beta,
        "per_ind_beta": per_ind_beta,
        "per_ind_beta_long": per_ind_beta_long,
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
df_per = df_per.melt(id_vars=["sub", "has_peak", "bg_loc"], value_vars=["per_low_beta", "per_high_beta", "per_all_beta", "per_ind_beta", "per_ind_beta_long"], var_name="model", value_name="per")
min_per = df_per.query("model == 'per_all_beta'").query("per == per.min()")  # results in sub rcs03l
df_per = df_per.query("sub != 'rcs03l'")  # remove the row with min per

# delete the rows where had_peak is True and model = per_ind_beta
df_per = df_per.query("not (has_peak == False and model == 'per_ind_beta')")

per_group = df_per.groupby("model")["per"].mean()
order_ = ["per_low_beta", "per_high_beta", "per_all_beta", "per_ind_beta", "per_ml"]
df_per.groupby("model")["per"].std()

l_per_ind = []
for sub in tqdm(ind_peaks_short.keys()):
    for symptom in ["pkg_bk", "pkg_dk", "pkg_tremor"]:
        pkg_label = df_all[df_all["sub"] == sub][symptom].values
        msk_ = ~np.isnan(pkg_label)
        pkg_label = pkg_label[msk_]
        
        power_sum = df_all[df_all["sub"] == sub][[f"ch_subcortex_welch_psd_{int(i)}_mean" for i in range(1, 115)]].apply(lambda x: 10**x).sum(axis=1).values
        power_sum = power_sum[msk_]
        for range_hz in range(3, 110):

            ind_band = df_all[df_all["sub"] == sub][f"ch_subcortex_welch_psd_{int(range_hz)}_mean"].apply(lambda x: 10**x).values
            ind_band = ind_band[msk_]
            
            if SPEARMAN:
                per_int_band = stats.spearmanr(ind_band / power_sum, pkg_label).correlation
            else:
                per_int_band = np.corrcoef(ind_band / power_sum, pkg_label)[0, 1]
            #P_val = stats.spearmanr(ind_band / power_sum, pkg_label).pvalue

            l_per_ind.append({
                "sub": sub,
                "per_ind_band": per_int_band,
                #"p_val": P_val,
                "band": range_hz,
                "symptom": symptom,
                #"peak_freq": ind_peaks_short[sub]
            })

df_per_ind = pd.DataFrame(l_per_ind)
df_plt = df_per_ind.query("band != '3-12' and band != '8-35' and band != '60-90'")
df_plt["has_peak"] = True
# if sub in rcs05r, rcs07r, rcs12r, rcs18l, rcs20l then has_peak is False
df_plt.loc[df_per_ind["sub"].isin(["rcs05r", "rcs07r", "rcs12r", "rcs18l", "rcs20l"]), "has_peak"] = False
df_plt["bg_loc"] = "STN"
subjects_GP = ["09l", "09r", "10l", "10r", "14l", "19l", "19r"]
# if sub in subjects_GP then bg_loc is GP
df_plt.loc[df_plt["sub"].str.contains("|".join(subjects_GP)), "bg_loc"] = "GP"
plt_var = "per_ind_band" 
colors = sns.color_palette("viridis", 2)
order = ["per_ind_beta", "per_all_beta"]

plt.figure(figsize=(9, 9))
plt.subplot(221)
df_plt_2 = df_per.query("bg_loc == 'STN'")
sns.boxplot(y="per", x="model", data=df_plt_2, hue="has_peak", showmeans=True, showfliers=False, palette="viridis", boxprops=dict(alpha=0.5), order=order)
sns.swarmplot(y="per", x="model", data=df_plt_2, hue="has_peak", dodge=True, alpha=0.3, color="black", order=order, legend=False)
plt.ylabel("Spearmann correlation coefficient") if SPEARMAN else plt.ylabel("Pearson's correlation coefficient")
plt.title("STN")

plt.subplot(223)
df_plt_2 = df_per.query("bg_loc == 'GP'")
sns.boxplot(y="per", x="model", data=df_plt_2, hue="has_peak", showmeans=True, showfliers=False, palette="viridis", boxprops=dict(alpha=0.5), order=order, color=colors[1])
sns.swarmplot(y="per", x="model", data=df_plt_2, hue="has_peak", dodge=True, alpha=0.3, order=order, legend=False, color="black")
plt.ylabel("Spearmann correlation coefficient") if SPEARMAN else plt.ylabel("Pearson's correlation coefficient")
plt.title("GP")

plt.subplot(222)
stn_ = df_plt.query("symptom == 'pkg_bk' and bg_loc == 'STN'").groupby("band")[plt_var].mean()
stn_var_ = df_plt.query("symptom == 'pkg_bk' and bg_loc == 'STN'").groupby("band")[plt_var].var()
plt.plot(stn_, color=colors[0])
plt.fill_between(stn_.index, stn_ - stn_var_, stn_ + stn_var_, alpha=0.3, color=colors[0])
plt.ylabel("Spearmann correlation coefficient") if SPEARMAN else plt.ylabel("Pearson's correlation coefficient")
plt.xlabel("Frequency [Hz]")
plt.title("STN")
plt.xlim([5, 35])
plt.ylim([-0.1, 0.5])

plt.subplot(224)
stn_ = df_plt.query("symptom == 'pkg_bk' and bg_loc == 'GP'").groupby("band")[plt_var].mean()
stn_var_ = df_plt.query("symptom == 'pkg_bk' and bg_loc == 'GP'").groupby("band")[plt_var].var()
plt.plot(stn_, color=colors[1])
plt.title("GP")
plt.fill_between(stn_.index, stn_ - stn_var_, stn_ + stn_var_, alpha=0.3, color=colors[1])
plt.ylabel("Spearmann correlation coefficient") if SPEARMAN else plt.ylabel("Pearson's correlation coefficient")

plt.xlim([5, 35])
plt.ylim([-0.1, 0.6])
plt.xlabel("Frequency [Hz]")
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "joint_fig_beta_Spearman.pdf"))

# Spearmans
# STN all_beta PEAK vs all all_beta NO_PEAK, p=0.81
# STN all_beta PEAK vs ind_beta PEAK PEAK, p=0.092
# GP all_beta PEAK vs ind_beta PEAK, p=0.34

# Pearsons
# 0.51
# 0.008
# 0.36

per_allbeta_peaks = df_per.query("model == 'per_all_beta' and has_peak == True and bg_loc=='STN'")["per"].values
per_allbeta_no_peaks = df_per.query("model == 'per_all_beta' and has_peak == False and bg_loc=='STN'")["per"].values
print(nm_stats.permutationTest(per_allbeta_peaks, per_allbeta_no_peaks, False, None, 5000))  # 0.87

subs_ind = df_per.query("model == 'per_ind_beta'")["sub"]
per_all_beta_peaks = df_per.query("model == 'per_all_beta' and bg_loc == 'STN' and has_peak == True").query("sub in @subs_ind")["per"].values
per_ind_beta_peaks = df_per.query("model == 'per_ind_beta' and bg_loc == 'STN' and has_peak == True").query("sub in @subs_ind")["per"].values
print(nm_stats.permutationTest_relative(per_all_beta_peaks, per_ind_beta_peaks, False, None, 5000))  # 0.016

per_allbeta_peaks = df_per.query("model == 'per_all_beta' and has_peak == True and bg_loc=='GP'")["per"].values
per_indbeta_peaks = df_per.query("model == 'per_ind_beta' and has_peak == True and bg_loc=='GP'")["per"].values
print(nm_stats.permutationTest(per_allbeta_peaks, per_indbeta_peaks, False, None, 5000))  # 0.35