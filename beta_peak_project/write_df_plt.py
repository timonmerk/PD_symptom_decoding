import pandas as pd
import seaborn as sb
import os
import numpy as np
import sys
from tqdm import tqdm
from scipy import stats

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


if __name__ == "__main__":


    PATH_ = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_std_10s_window_length_all_ch/all_merged.csv"
    PATH_ = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_std_10s_window_length/all_merged_with_condition.csv"
    ch_used = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/ch_used_per_sub.csv"
    PATH_OUT = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per"

    PATH_COORD = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/paper_per/ind_ch'
    df_coords = pd.read_csv(os.path.join(PATH_COORD, 'df_per_ind_all_coords.csv'))

    df = pd.read_csv(PATH_, index_col=0)
    df = df[df["condition"] == "stim_off"]
    df = df.drop(columns=["condition"])
    df["pkg_dt"] = pd.to_datetime(df["pkg_dt"], utc=True).dt.tz_convert("US/Pacific")
    df_ch_used = pd.read_csv(ch_used, index_col=0)
    per_l = []

    for sub in tqdm(df_ch_used["sub"].unique()):
        ch_names_orig = df_ch_used[df_ch_used["sub"] == sub].iloc[0, :4].values
        ch_names = df_ch_used.columns[:4]
        ch_names = [ch for ch in ch_names if "ch_subcortex" in ch]
        ch_names_orig = [ch for ch in ch_names_orig if ch.startswith("0-") or ch.startswith("1-") or ch.startswith("2-")]
        for ch_idx, ch in enumerate(ch_names):
            df_sub = df[df["sub"] == sub].copy()
            df_sub["pkg_dt"] = pd.to_datetime(df_sub["pkg_dt"])
            df_sub["h"] = df_sub["pkg_dt"].dt.hour
            df_sub_ch = df_sub[[c for c in df_sub.columns if c.startswith(ch) and "psd" in c or c == "h" or c == "pkg_bk"]].copy()
            df_sub_ch = df_sub_ch.query("h >= 8 and h <= 20")

            #for label in ["pkg_dk", "pkg_bk", "pkg_tremor"]:
            label = "pkg_bk"
            X = df_sub_ch.copy()
            y = df_sub_ch[label].copy()

            idx_nan = X.isna().any(axis=1)
            X = X[~idx_nan]
            y = y[np.array(~idx_nan)]
            idx_nan = y.isna()
            X = X[~idx_nan]
            y = y[np.array(~idx_nan)]

            power_sum = X[[f"{ch}_welch_psd_{int(i)}_mean" for i in range(1, 125)]].apply(lambda x: 10**x).sum(axis=1).values
            ind_beta = X[f"{ch}_welch_psd_{ind_peaks_short[sub]}_mean"].apply(lambda x: 10**x).values

            all_beta = X[[f"{ch}_welch_psd_{i}_mean" for i in range(8, 31)]].apply(
                lambda x: 10**x
            ).mean(axis=1).values

            #per_ind = np.corrcoef(y, ind_beta / power_sum)[0, 1]
            #per_all = np.corrcoef(y, all_beta / power_sum)[0, 1]
            per_ind = stats.spearmanr(y, ind_beta / power_sum).correlation
            per_all = stats.spearmanr(y, all_beta / power_sum).correlation

            try:
                xyz = df_coords.query(f"sub == '{sub}' and ch_orig == '{ch_names_orig[ch_idx]}'").iloc[0][["x", "y", "z"]]
            except:
                continue
            per_l.append({
                "sub": sub,
                "ch": ch,
                "ch_orig": ch_names_orig[ch_idx],
                "per_ind": per_ind,
                "per_all": per_all,
                "x": xyz["x"],
                "y": xyz["y"],
                "z": xyz["z"],
            })

    df_per = pd.DataFrame(per_l)
    df_per.to_csv("beta_peak_project/per_combined_coords_beta.csv", index=False)