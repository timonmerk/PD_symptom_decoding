import pandas as pd
import numpy as np


subs_GP = ["rcs09l", "rcs09r", "rcs10l", "rcs10r", "rcs14r", "rcs19l", "rcs19r"]
patients_without_peak = [ "rcs08l", "rcs10l", "rcs12r", "rcs18l"]

def get_per_loc(label, loc):
    if loc == "ML_ECOG_SC":
        df = pd.read_csv("publication_figures/df_comp_beta_ml_ecog_stn.csv")
        return df.query("label == @label and type == 'corr_pr'")["value"].values
    elif loc == "ML_ECOG":
        df = pd.read_csv("publication_figures/df_comp_beta_ml_ecog.csv")
        return df.query("label == @label and type == 'corr_pr'")["value"].values
    elif loc == "ML_STN":
        df = pd.read_csv("publication_figures/df_comp_beta_ml_stn.csv")
        return df.query("label == @label and type == 'corr_pr'")["value"].values
    elif loc == "Beta_SC":
        df = pd.read_csv("publication_figures/df_comp_beta_ml_stn.csv")
        return df.query("label == @label and type == 'corr_ind'")["value"].values
    elif loc == "Beta_STN":
        df = pd.read_csv("publication_figures/df_comp_beta_ml_stn.csv")
        df["region"] = "STN"
        df.loc[df["sub"].isin(subs_GP), "region"] = "GP"
        return df.query("label == @label and type == 'corr_ind' and region == 'STN'")["value"].values
    elif loc == "Beta_STN_wo_peak":
        df = pd.read_csv("publication_figures/df_comp_beta_ml_stn.csv")
        df["region"] = "STN"
        df.loc[df["sub"].isin(subs_GP), "region"] = "GP"
        df["peak_present"] = 1
        df.loc[df["sub"].isin(patients_without_peak), "peak_present"] = 0
        return df.query("label == @label and type == 'corr_ind' and region == 'STN' and peak_present == 0")["value"].values
    elif loc == "Beta_GP":
        df = pd.read_csv("publication_figures/df_comp_beta_ml_stn.csv")
        df["region"] = "STN"
        df.loc[df["sub"].isin(subs_GP), "region"] = "GP"
        return df.query("label == @label and type == 'corr_ind' and region == 'GP'")["value"].values
    elif loc == "Beta_GP_wo_peak":
        df = pd.read_csv("publication_figures/df_comp_beta_ml_stn.csv")
        df["region"] = "STN"
        df.loc[df["sub"].isin(subs_GP), "region"] = "GP"
        df["peak_present"] = 1
        df.loc[df["sub"].isin(patients_without_peak), "peak_present"] = 0
        return df.query("label == @label and type == 'corr_ind' and region == 'GP' and peak_present == 0")["value"].values

locs = ["ML_ECOG_SC", "ML_ECOG", "ML_STN", "Beta_SC", "Beta_STN", "Beta_STN_wo_peak", "Beta_GP", "Beta_GP_wo_peak"]
for label in ["pkg_bk", "pkg_dk", "pkg_tremor"]:
    print(f"\n\n{label}")
    for loc in locs:
        res = get_per_loc(label, loc)
        print(f"{label} {loc}: {np.nanmean(res):.2f} ± {np.nanstd(res):.2f}")
