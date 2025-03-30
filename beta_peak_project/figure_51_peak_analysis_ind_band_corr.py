import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import pickle
from tqdm import tqdm


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

COMPUTE_EACH_BAND = True
l_per = []
for sub in tqdm(ind_peaks_short.keys()):
    for symptom in ["pkg_bk", "pkg_dk", "pkg_tremor"]:
        pkg_label = df_all[df_all["sub"] == sub][symptom].values
        msk_ = ~np.isnan(pkg_label)
        pkg_label = pkg_label[msk_]
        
        if COMPUTE_EACH_BAND:
            power_sum = df_all[df_all["sub"] == sub][[f"ch_subcortex_welch_psd_{int(i)}_mean" for i in range(1, 115)]].apply(lambda x: 10**x).sum(axis=1).values
            power_sum = power_sum[msk_]
            for range_hz in range(3, 110):

                #ind_band = df_all[df_all["sub"] == sub][[f"ch_subcortex_welch_psd_{int(i)}_mean" for i in range(int(range_hz-2.5), int(range_hz+2.5))]].mean(axis=1).values
                ind_band = df_all[df_all["sub"] == sub][f"ch_subcortex_welch_psd_{int(range_hz)}_mean"].apply(lambda x: 10**x).values
                ind_band = ind_band[msk_]
                
                #per_int_band = np.corrcoef(ind_band / power_sum, pkg_label)[0, 1]
                per_int_band = stats.spearmanr(ind_band / power_sum, pkg_label).correlation
                P_val = stats.spearmanr(ind_band / power_sum, pkg_label).pvalue

                l_per.append({
                    "sub": sub,
                    "per_ind_band": per_int_band,
                    "p_val": P_val,
                    "band": range_hz,
                    "symptom": symptom,
                    #"peak_freq": ind_peaks_short[sub]
                })

        range_1 = [3, 12]
        ind_low_fband = df_all[df_all["sub"] == sub][[f"ch_subcortex_welch_psd_{int(i)}_mean" for i in range(int(range_1[0]), int(range_1[1]))]].mean(axis=1).values

        range_2 = [60, 90]
        ind_high_fband = df_all[df_all["sub"] == sub][[f"ch_subcortex_welch_psd_{int(i)}_mean" for i in range(int(range_2[0]), int(range_2[1]))]].mean(axis=1).values

        range_3 = [8, 35]
        all_beta = df_all[df_all["sub"] == sub][[f"ch_subcortex_welch_psd_{int(i)}_mean" for i in range(int(range_3[0]), int(range_3[1]))]].mean(axis=1).values

        ind_low_fband = ind_low_fband[msk_]
        per_lfband = np.corrcoef(ind_low_fband, pkg_label)[0, 1]

        ind_high_fband = ind_high_fband[msk_]
        per_hfband = np.corrcoef(ind_high_fband, pkg_label)[0, 1]

        all_bea = all_beta[msk_]
        per_beta = np.corrcoef(all_bea, pkg_label)[0, 1]

        # l_per.append({
        #     "sub": sub,
        #     "per_ind_band": per_lfband,
        #     "band": "3-12",
        #     "symptom": symptom,
        # })

        # l_per.append({
        #     "sub": sub,
        #     "per_ind_band": per_hfband,
        #     "band": "60-90",
        #     "symptom": symptom,
        # })

        # l_per.append({
        #     "sub": sub,
        #     "per_ind_band": per_beta,
        #     "band": "8-35",
        #     "symptom": symptom,
        # })

df_per = pd.DataFrame(l_per)
df_plt = df_per.query("band != '3-12' and band != '8-35' and band != '60-90'")
df_plt["has_peak"] = True
# if sub in rcs05r, rcs07r, rcs12r, rcs18l, rcs20l then has_peak is False
df_plt.loc[df_per["sub"].isin(["rcs05r", "rcs07r", "rcs12r", "rcs18l", "rcs20l"]), "has_peak"] = False
df_plt["bg_loc"] = "STN"
subjects_GP = ["09l", "09r", "10l", "10r", "14l", "19l", "19r"]
# if sub in subjects_GP then bg_loc is GP
df_plt.loc[df_plt["sub"].str.contains("|".join(subjects_GP)), "bg_loc"] = "GP"


colors = sns.color_palette("viridis", 4)
plt.figure(figsize=(4, 6))
plt_var = "per_ind_band" # "p_val" 
# all_mean = df_plt.query("symptom == 'pkg_bk'").groupby("band")[plt_var].mean()
# all_var = df_plt.query("symptom == 'pkg_bk'").groupby("band")[plt_var].var()
# plt.plot(all_mean, label="ALL", color=colors[0])
# plt.fill_between(all_mean.index, all_mean - all_var, all_mean + all_var, alpha=0.3, color=colors[0])

plt.subplot(211)
stn_ = df_plt.query("symptom == 'pkg_bk' and bg_loc == 'STN'").groupby("band")[plt_var].mean()
stn_var_ = df_plt.query("symptom == 'pkg_bk' and bg_loc == 'STN'").groupby("band")[plt_var].var()
plt.plot(stn_, label="STN with peak", color=colors[1])
plt.fill_between(stn_.index, stn_ - stn_var_, stn_ + stn_var_, alpha=0.3, color=colors[1])
plt.ylabel("Spearmann correlation coefficient")
plt.xlim([5, 35])

plt_var = "p_val" 
plt.subplot(212)
stn_ = df_plt.query("symptom == 'pkg_bk' and bg_loc == 'STN'").groupby("band")[plt_var].mean()
stn_var_ = df_plt.query("symptom == 'pkg_bk' and bg_loc == 'STN'").groupby("band")[plt_var].var()
plt.plot(stn_, label="STN with peak", color=colors[1])
plt.fill_between(stn_.index, stn_ - stn_var_, stn_ + stn_var_, alpha=0.3, color=colors[1])
plt.ylabel("Spearmann correlation coefficient")

# stn_without_peak = df_plt.query("symptom == 'pkg_bk' and has_peak == False and bg_loc == 'STN'").groupby("band")[plt_var].mean()
# stn_without_peak_var = df_plt.query("symptom == 'pkg_bk' and has_peak == False and bg_loc == 'STN'").groupby("band")[plt_var].var()
# plt.plot(stn_without_peak, label="STN without peak", color=colors[2])
# plt.fill_between(stn_without_peak.index, stn_without_peak - stn_without_peak_var, stn_without_peak + stn_without_peak_var, alpha=0.3, color=colors[2])

# gp = df_plt.query("symptom == 'pkg_bk' and bg_loc == 'GP'").groupby("band")[plt_var].mean()
# gp_var = df_plt.query("symptom == 'pkg_bk' and bg_loc == 'GP'").groupby("band")[plt_var].var()
# plt.plot(gp, label="GP", color=colors[3])
# plt.fill_between(gp.index, gp - gp_var, gp + gp_var, alpha=0.3, color=colors[3])
plt.ylabel("p-value")
plt.xlabel("Frequency [Hz]")
plt.xlim([5, 35])
plt.ylim([0, 0.05])
plt.savefig(os.path.join(PATH_FIGURES, "per_ind_band_all.pdf"))


df_per["per_ind_band_abs"] = np.abs(df_per["per_ind_band"])

#df_per_best = df_per.groupby(["sub", "symptom", "band"])["per_ind_band_abs"].max().reset_index()
# get the range_hz with the best performance
#df_per_best = df_per_best.merge(df_per, on=["sub", "symptom", "per_ind_band_abs"], how="left")

# rename df_per_best "per_ind_band" to "per"
df_per = df_per.rename(columns={"per_ind_band": "per"})
if COMPUTE_EACH_BAND:
    df_per_ml["band"] = None
# drop include_night column
df_per_ml = df_per_ml.drop("include_night", axis=1)
df_per_ml = df_per_ml.drop("CLASSIFICATION", axis=1)
df_per_ml = df_per_ml.drop("Unnamed: 0", axis=1)

df_per_best = df_per.drop("per_ind_band_abs", axis=1)
df_per_best = df_per_best.rename(columns={"symptom": "pkg_label"})
#df_per_best["model"] = "per_ind_best_band"
df_per_ml["band"] = "per_ml"

df_comb = pd.concat([df_per_best, df_per_ml], axis=0)
df_comb["per_abs"] = np.abs(df_comb["per"]) 

from py_neuromodulation import nm_stats
nm_stats.permutationTest_relative(
    df_comb.query("band == '60-90' and pkg_label == 'pkg_bk'")["per_abs"].values,
    df_comb.query("band == 'per_ml' and pkg_label == 'pkg_bk'")["per_abs"].values,
    False,
    5000
)

plt.figure()
sns.boxplot(data=df_comb, x="pkg_label", y="per_abs", hue="band", showfliers=False, showmeans=True, hue_order=["3-12", "8-35", "60-90", "per_ml"], palette="viridis")
sns.swarmplot(data=df_comb, x="pkg_label", y="per_abs", hue="band", color=".25", dodge=True, legend=False, hue_order=["3-12", "8-35", "60-90", "per_ml"], alpha=0.4)
plt.ylabel("Performance [r]")
plt.xlabel("Symptom")
plt.show(block=True)


plt.figure()
plt.subplot(121)
sns.boxplot(data=df_comb, x="pkg_label", y="per_abs", hue="model", showfliers=False, showmeans=True)
sns.swarmplot(data=df_comb, x="pkg_label", y="per_abs", hue="model", color=".25", dodge=True, legend=False)
plt.ylabel("Performance [r]")
plt.xlabel("Symptom")
plt.title("Performance of the best band")

plt.subplot(122)
sns.boxplot(data=df_comb.query("model != 'per_ml'"), x="pkg_label", y="band", showfliers=False, showmeans=True)
sns.swarmplot(data=df_comb.query("model != 'per_ml'"), x="pkg_label", y="band", color=".25", legend=False)
plt.ylabel("Frequency [Hz]")
plt.xlabel("Symptom")
plt.title("Frequency of the best band")
plt.show(block=True)

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
df_per = df_per.query("not (has_peak == True and model == 'per_ind_beta')")
