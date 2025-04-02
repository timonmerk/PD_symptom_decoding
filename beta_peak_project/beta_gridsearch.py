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
df_all = df_all.query("h >= 8 and h <= 20")

list(df_all.columns)

def compute_sub(sub):
    l_per = []
    for symptom in ["pkg_bk", ]: #  "pkg_dk", "pkg_tremor"
        pkg_label = df_all[df_all["sub"] == sub][symptom].values
        msk_ = ~np.isnan(pkg_label)
        pkg_label = pkg_label[msk_]
        
        power_sum = df_all[df_all["sub"] == sub][[f"ch_subcortex_welch_psd_{int(i)}_mean" for i in range(1, 115)]].apply(lambda x: 10**x).sum(axis=1).values
        power_sum = power_sum[msk_]
        for range_low in range(5, 34):
            for range_high in range(6, 36):
                range_idx = np.arange(range_low, range_high)
                ind_band = df_all[df_all["sub"] == sub][[f"ch_subcortex_welch_psd_{int(i)}_mean" for i in range(range_low, range_high)]].apply(lambda x: 10**x).mean(axis=1).values

                ind_band = ind_band[msk_]
            
                per_int_band = np.corrcoef(ind_band / power_sum, pkg_label)[0, 1]
                #res = stats.spearmanr(ind_band / power_sum, pkg_label)
                #per_int_band = res.correlation
                #P_val = res.pvalue

                l_per.append({
                    "sub": sub,
                    "per_ind_band": per_int_band,
                    #"p_val": P_val,
                    "range" : f"{range_low}-{range_high}",
                    "symptom": symptom,
                    "range_low": range_low,
                    "range_high": range_high
                })
    return l_per

from joblib import Parallel, delayed

subs = df_all["sub"].unique()
l_per = Parallel(n_jobs=len(subs))(delayed(compute_sub)(sub) for sub in tqdm(subs))
l_per = [item for sublist in l_per for item in sublist]
df_gs_pre = pd.DataFrame(l_per)

per_mean_gs = []
for sub_test in subs:
    low_range = []
    high_range = []
    for sub in subs:
        if sub == sub_test:
            continue
        df_sub = df_gs_pre.query(f"sub == '{sub}'")
        df_best = df_sub.query("per_ind_band == per_ind_band.max()").iloc[0]
        low_range.append(df_best["range_low"])
        high_range.append(df_best["range_high"])
    low_range = np.array(low_range)
    high_range = np.array(high_range)

    low_mean = low_range.mean()
    high_mean = high_range.mean()
    pkg_label = df_all[df_all["sub"] == sub_test]["pkg_bk"].values
    msk_ = ~np.isnan(pkg_label)
    pkg_label = pkg_label[msk_]
    
    power_sum = df_all[df_all["sub"] == sub_test][[f"ch_subcortex_welch_psd_{int(i)}_mean" for i in range(1, 115)]].apply(lambda x: 10**x).sum(axis=1).values
    power_sum = power_sum[msk_]
    ind_band = df_all[df_all["sub"] == sub_test][[f"ch_subcortex_welch_psd_{int(i)}_mean" for i in range(int(low_mean), int(high_mean))]].apply(lambda x: 10**x).mean(axis=1).values
    ind_band = ind_band[msk_]
    #per_int_band = np.corrcoef(ind_band / power_sum, pkg_label)[0, 1]
    res = stats.spearmanr(ind_band / power_sum, pkg_label)
    per_int_band = res.correlation
    P_val = res.pvalue

    per_mean_gs.append({
        "sub": sub_test,
        "per_ind_band": per_int_band,
        "p_val": P_val,
        "range" : f"{low_mean}-{high_mean}",
        "symptom": "pkg_bk",
        "range_low": low_mean,
        "range_high": high_mean
    })

df_per_mean_gs = pd.DataFrame(per_mean_gs)
df_per_mean_gs.to_csv(os.path.join(PATH_PER, "grid_search_mean_sub.csv"), index=False)


def compute_sub_cv(sub_test, symptom="pkg_bk"):
    l_per = []

    pkg_label = df_all[df_all["sub"] != sub_test][symptom].values
    msk_ = ~np.isnan(pkg_label)
    pkg_label = pkg_label[msk_]
    
    power_sum = df_all[df_all["sub"] != sub_test][[f"ch_subcortex_welch_psd_{int(i)}_mean" for i in range(1, 115)]].apply(lambda x: 10**x).sum(axis=1).values
    power_sum = power_sum[msk_]
    for range_low in range(5, 34):
        for range_high in range(6, 36):
            range_idx = np.arange(range_low, range_high)
            ind_band = df_all[df_all["sub"] != sub_test][[f"ch_subcortex_welch_psd_{int(i)}_mean" for i in range(range_low, range_high)]].apply(lambda x: 10**x).mean(axis=1).values

            ind_band = ind_band[msk_]
        
            #per_int_band = np.corrcoef(ind_band / power_sum, pkg_label)[0, 1]
            res = stats.spearmanr(ind_band / power_sum, pkg_label)
            per_int_band = res.correlation
            P_val = res.pvalue

            l_per.append({
                "sub": sub_test,
                "per_ind_band": per_int_band,
                "p_val": P_val,
                "range" : f"{range_low}-{range_high}",
                "symptom": symptom,
                "range_low": range_low,
                "range_high": range_high
            })
    df_per_cv = pd.DataFrame(l_per)
    # get row of mac per_ind_band
    df_per_cv = df_per_cv.query("per_ind_band == per_ind_band.max()")
    best_r_low = df_per_cv["range_low"].values[0]
    best_r_high = df_per_cv["range_high"].values[0]

    pkg_label = df_all[df_all["sub"] == sub_test][symptom].values
    msk_ = ~np.isnan(pkg_label)
    pkg_label = pkg_label[msk_]
    
    power_sum = df_all[df_all["sub"] == sub_test][[f"ch_subcortex_welch_psd_{int(i)}_mean" for i in range(1, 115)]].apply(lambda x: 10**x).sum(axis=1).values
    power_sum = power_sum[msk_]
    ind_band = df_all[df_all["sub"] == sub_test][[f"ch_subcortex_welch_psd_{int(i)}_mean" for i in range(best_r_low, best_r_high)]].apply(lambda x: 10**x).mean(axis=1).values
    ind_band = ind_band[msk_]
    #per_int_band = np.corrcoef(ind_band / power_sum, pkg_label)[0, 1]
    res = stats.spearmanr(ind_band / power_sum, pkg_label)
    per_int_band = res.correlation
    P_val = res.pvalue

    res_return = {
        "sub": sub_test,
        "per_ind_band": per_int_band,
        "p_val": P_val,
        "range" : f"{best_r_low}-{best_r_high}",
        "symptom": symptom,
        "range_low": best_r_low,
        "range_high": best_r_high
    }

    return res_return

#compute_sub_cv(subs[0])
from joblib import Parallel, delayed
l_per = Parallel(n_jobs=len(subs))(delayed(compute_sub_cv)(sub) for sub in tqdm(subs))
df_per_cv = pd.DataFrame(l_per)
df_per_cv.to_csv(os.path.join(PATH_PER, "grid_search_cv_bands.csv"), index=False)




df_per = pd.DataFrame(l_per)

df_per["has_peak"] = True
# if sub in rcs05r, rcs07r, rcs12r, rcs18l, rcs20l then has_peak is False
df_per.loc[df_per["sub"].isin(["rcs05r", "rcs07r", "rcs12r", "rcs18l", "rcs20l"]), "has_peak"] = False
df_per["bg_loc"] = "STN"
subjects_GP = ["09l", "09r", "10l", "10r", "14l", "19l", "19r"]
# if sub in subjects_GP then bg_loc is GP
df_per.loc[df_per["sub"].str.contains("|".join(subjects_GP)), "bg_loc"] = "GP"

df_per = df_per.query("bg_loc == 'STN'")

# show in an image the performances, x axis range-Low y axis range_high, color per performance
plt.figure(figsize=(8, 8))
for idx_sym, symptom in enumerate(["pkg_bk", "pkg_dk", "pkg_tremor"]):
    for idx_plt, plt_val in enumerate(["per_ind_band", "p_val"]):
        plt.subplot(3, 2, idx_sym * 2 + idx_plt + 1)
        df_plt = df_per.query(f"symptom == @symptom")[["range_low", "range_high", "per_ind_band", "p_val", "sub"]]
        # remove nans
        df_plt = df_plt.dropna()

        df_plt = df_plt.groupby(["range_low", "range_high"])[["per_ind_band", "p_val"]].mean().reset_index()
        
        color_ = df_plt[plt_val].values
        if plt_val == "p_val":
            color_ = np.log10(color_)
        plt.scatter(df_plt["range_low"], df_plt["range_high"], c=color_, cmap="jet", s=50, marker="s")
        plt.xlabel("Range low [Hz]")
        plt.ylabel("Range high [Hz]")
        if plt_val == "per_ind_band":
            min_low, min_high = df_plt.loc[df_plt[plt_val].idxmax(), ["range_low", "range_high"]]
            plt_title = "Corr. coef."
            plt.title(f"{plt_title}\n{symptom} max: {int(min_low)}-{int(min_high)}\n corr.: {df_plt[plt_val].max():.2f}")
        else:
            df_plt_ = df_plt.query("per_ind_band > 0")
            min_low, min_high = df_plt_.loc[df_plt_[plt_val].idxmin(), ["range_low", "range_high"]]
            plt_title = "p-value"
            plt.title(f"{plt_title}\n{symptom} min: {int(min_low)}-{int(min_high)}\n p-val: {df_plt_[plt_val].min():.2f}")
        plt.colorbar()
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "grid_search_bands.pdf"))
plt.show()


