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

ind_peaks_long = {
    "rcs02l" : 20,
    "rcs02r" : 18,
    "rcs03l" : 13.5,
    "rcs05l" : 26,
    "rcs05r" : 26,
    "rcs06l" : 28,
    "rcs06r" : 18,
    "rcs07l" : 13,
    "rcs07r" : 8, # 
    "rcs08l" : 25,  # none
    "rcs08r" : 27,
    "rcs09l" : 24,
    "rcs09r" : 23,
    "rcs10l" : 27, # none
    "rcs10r" : 29,
    "rcs11l" : 27,
    "rcs11r" : 25,
    "rcs12l" : 28,
    "rcs12r" : 28, # none
    "rcs14l" : 25,
    "rcs15l" : 22,
    "rcs15r" : 18,
    "rcs17l" : 27,
    "rcs17r" : 29,
    "rcs18l" : 23, # none
    "rcs18r" : 23,
    "rcs19l" : 10,
    "rcs19r" : 22,
    "rcs20l" : 17,
    "rcs20r" : 17,
}


#patients_without_peak = ["rcs05r", "rcs07r", "rcs12l", "rcs18l", "rcs20l"]
patients_without_peak = [ "rcs08l", "rcs10l", "rcs12r", "rcs18l"]

PATH_FIGURES = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf/figures_paper'
PATH_PER = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/paper_per"

df_features = pd.read_csv("/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_std_10s_window_length/all_merged_preprocessed_with_condition.csv")
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
for label in ["pkg_bk", "pkg_dk", "pkg_tremor"]:

    df_features = df_features_orig.copy()
    mask = ~df_features[label].isnull()
    df_features = df_features[mask].copy()
    df_features = df_features.drop(columns=df_features.columns[df_features.isnull().all()])
    df_features = df_features.replace([np.inf, -np.inf], np.nan)
    df_features = df_features.dropna(axis=1)

    file = f"LOHO_main_{label}_CLASS_False_loc_ecog_stn_nonorm_withpsd.pkl"
    #file = f"LOHO_main_{label}_CLASS_False_loc_ecog_nonorm_withpsd.pkl"  # select here for ECoG only
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

        power_sum = df_sub[[f"ch_subcortex_welch_psd_{int(i)}_mean" for i in range(1, 115)]].apply(lambda x: 10**x).sum(axis=1).values

        if sub not in patients_without_peak:
            ind_band = df_sub[[f"ch_subcortex_welch_psd_{int(i)}_mean" for i in range(int(ind_peaks_long[sub]-2.5), int(ind_peaks_long[sub]+2.5))]].apply(lambda x: 10**x).mean(axis=1).values
        else:
            ind_band = df_sub[[f"ch_subcortex_welch_psd_{int(i)}_mean" for i in range(int(19-2.5), int(19+2.5))]].apply(lambda x: 10**x).mean(axis=1).values
        #ind_band = stats.zscore(ind_band)
        #pr_ = stats.zscore(pr_)
        #true_ = stats.zscore(true_)

        if CORR_SPEARMANS is False:
            corr_ind = np.corrcoef(ind_band / power_sum, true_)[0, 1]
            corr_pr = np.corrcoef(pr_, true_)[0, 1]
        else:
            corr_ind = stats.spearmanr(ind_band  / power_sum, true_).correlation
            corr_pr = stats.spearmanr(pr_, true_).correlation

        df_comp.append({
            "sub" : sub,
            "corr_ind" : corr_ind if sub not in patients_without_peak else np.nan,
            "corr_pr" : corr_pr,
            "label": label
        })

        if PLT_:
            plt.figure(figsize=(10, 3))
            plt.plot(pr_, label="Prediction", color="red")
            plt.plot(true_, label="True", color="blue")
            plt.plot(ind_band, label="Band", color="green")
            plt.legend()
            
            plt.title(f"{sub}, corr_beta: {corr_ind:.2f}, corr_ml: {corr_pr:.2f}")
            
            pdf_pages.savefig()
            plt.close()
    if PLT_:
        pdf_pages.close()

df_comp = pd.DataFrame(df_comp)
# set peak_present based on patients_without_peak
df_comp["peak_present"] = df_comp["sub"].apply(lambda x: 0 if x in patients_without_peak else 1)

df_comp["corr_ind_abs"] = np.abs(df_comp["corr_ind"])

df_comp = df_comp.melt(id_vars=["sub", "label", "peak_present"], value_vars=["corr_ind", "corr_ind_abs", "corr_pr"],
                       var_name="type", value_name="value")
order_ = ["pkg_bk", "pkg_dk", "pkg_tremor"]

df_comp.to_csv(os.path.join("publication_figures", f"df_comp_beta_ml.csv"), index=False)
