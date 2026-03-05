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



# get the DBS electrode performances grouped by peak present
df = pd.read_csv("publication_figures/df_comp_beta_ml_stn.csv")
df["region"] = "STN"
df.loc[df["sub"].isin(subs_GP), "region"] = "GP"
df["peak_present"] = 1
df.loc[df["sub"].isin(patients_without_peak), "peak_present"] = 0
for type in ["corr_ind", "corr_pr"]:
    for label in ["pkg_bk", "pkg_dk", "pkg_tremor"]:
        for peak_present in [1, 0]:
            res = df.query("type == @type and label == @label and peak_present == @peak_present")["value"].values
            print(f"{type} {label} peak_present {peak_present}: {np.nanmean(res):.2f} ± {np.nanstd(res):.2f}")
        # peak_present both
        res = df.query("type == @type and label == @label")["value"].values
        print(f"{type} {label} peak_present both: {np.nanmean(res):.2f} ± {np.nanstd(res):.2f}")

# get the above two in one dataframe, s.t. the value is mean pm std
for label in ["pkg_bk", "pkg_dk", "pkg_tremor"]:
    res = df.query("type == 'corr_ind' and label == @label and value <= 0")
    subs = res["sub"].unique()
    # get the mean and std for type corr_pr for those patients and those labels
    res = df.query("type == 'corr_pr' and label == @label and sub in @subs")["value"].values
    print(f"{label} corr_pr for patients with negative corr_ind: {np.nanmean(res):.2f} ± {np.nanstd(res):.2f}")

    
# Label	Beta peak present	ML Pearson corr. (r)	Beta Pearson corr. (r)
# Bradykinesia	True	0.51 ± 0.22	0.26 ± 0.32
# 	False	0.43 ± 0.29	0.50 ± 0.14
# 	Both	0.50 ± 0.23	0.29 ± 0.31
# Tremor	True	0.29 ± 0.31	0.24 ± 0.42
# 	False	0.19 ± 0.27	0.24 ± 0.24
# 	Both	0.28 ± 0.31	0.24 ± 0.40
# Dyskinesia	True	0.43 ± 0.28	-0.29 ± 0.28
# 	False	0.39 ± 0.36	-0.34 ± 0.04
# 	Both	0.43 ± 0.29	-0.30 ± 0.26

# Table 3: DBS electrode leave one hemisphere out cross-validation and beta peak performances. Performances are shown as Pearson correlation coefficient for regression. For n=4 hemispheres out of n=40 a beta peak was not present.

