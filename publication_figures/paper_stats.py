import pickle
import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
from scipy import stats
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
import os
from py_neuromodulation import nm_stats


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

subs_GP = ["rcs09l", "rcs09r", "rcs10l", "rcs10r", "rcs14r", "rcs19l", "rcs19r"]

PATH_PER = "publication_figures"
LOAD_ECOG = False

if LOAD_ECOG is False:
    df_comp_withnight = pd.read_csv(os.path.join(PATH_PER, "df_comp_beta_ml_incl_night.csv"))
else:
    #df_comp_withnight = pd.read_csv(os.path.join(PATH_PER, "df_comp_beta_ml_incl_night_ecog.csv"))
    print("")

if LOAD_ECOG is False:
    df_comp = pd.read_csv(os.path.join(PATH_PER, "df_comp_beta_ml.csv"))
else:
    df_comp = pd.read_csv(os.path.join(PATH_PER, "df_comp_beta_ml_ecog.csv"))

df_comp["loc"] = "STN"
df_comp.loc[df_comp["sub"].isin(subs_GP), "loc"] = "GP"

nm_stats.permutationTest(
    df_comp.query("type == 'corr_pr' and label == 'pkg_bk' and loc == 'STN'")["value"].values,
    df_comp.query("type == 'corr_pr' and label == 'pkg_bk' and loc == 'GP'")["value"].values,
    False, None, 5000
)

nm_stats.permutationTest(
    df_comp.query("type == 'corr_pr' and label == 'pkg_dk' and loc == 'STN'")["value"].values,
    df_comp.query("type == 'corr_pr' and label == 'pkg_dk' and loc == 'GP'")["value"].values,
    False, None, 5000
)

nm_stats.permutationTest(
    df_comp.query("type == 'corr_pr' and label == 'pkg_tremor' and loc == 'STN'")["value"].values,
    df_comp.query("type == 'corr_pr' and label == 'pkg_tremor' and loc == 'GP'")["value"].values,
    False, None, 5000
)


subs_smaller = df_comp.query("label == 'pkg_bk' and type == 'corr_ind' and value<0")["sub"]
vals_ind = df_comp.query("label == 'pkg_bk' and type == 'corr_ind' and value<0")["value"].values
np.mean(vals_ind)  # -0.11
np.std(vals_ind)   # 0.11
vals_pr = df_comp.query("label == 'pkg_bk' and type == 'corr_pr' and sub in @subs_smaller")["value"].values

vals_beta_tremor = df_comp.query("label == 'pkg_tremor' and type == 'corr_ind'")["value"].values
vals_beta_tremor_nonan = vals_beta_tremor[~np.isnan(vals_beta_tremor)]

nm_stats.permutationTest(
    vals_beta_tremor_nonan,
    np.zeros(vals_beta_tremor_nonan.shape[0]),
    False, None, 5000
)

nm_stats.permutationTest_relative(
    df_comp.query("label == 'pkg_bk' and type == 'corr_pr' and peak_present == 1")["value"].values,
    df_comp.query("label == 'pkg_bk' and type == 'corr_ind' and peak_present == 1")["value"].values,
    False, None, 5000
)

nm_stats.permutationTest_relative(
    df_comp.query("label == 'pkg_dk' and type == 'corr_pr' and peak_present == 1")["value"].values,
    df_comp.query("label == 'pkg_dk' and type == 'corr_ind' and peak_present == 1")["value"].values*-1,
    False, None, 5000
)

nm_stats.permutationTest_relative(
    df_comp.query("label == 'pkg_tremor' and type == 'corr_pr' and peak_present == 1")["value"].values,
    df_comp.query("label == 'pkg_tremor' and type == 'corr_ind' and peak_present == 1")["value"].values,
    False, None, 5000
)

# compare here the days vs whole-day 

np.nanmean(df_comp.query("label == 'pkg_bk' and type == 'corr_pr' and peak_present == 0")["value"].values)
np.nanstd(df_comp.query("label == 'pkg_bk' and type == 'corr_pr' and peak_present == 0")["value"].values)

nm_stats.permutationTest(
    df_comp.query("label == 'pkg_bk' and type=='corr_pr' and peak_present == 0")["value"].values,
    np.zeros(df_comp.query("label == 'pkg_bk' and type=='corr_pr' and peak_present == 0")["value"].values.shape[0]),
    False, None, 5000
)

np.nanmean(df_comp.query("label == 'pkg_tremor' and type == 'corr_pr' and peak_present == 0")["value"].values)
np.nanstd(df_comp.query("label == 'pkg_tremor' and type == 'corr_pr' and peak_present == 0")["value"].values)

nm_stats.permutationTest(
    df_comp.query("label == 'pkg_tremor' and type=='corr_pr' and peak_present == 0")["value"].values,
    np.zeros(df_comp.query("label == 'pkg_tremor' and type=='corr_pr' and peak_present == 0")["value"].values.shape[0]),
    False, None, 5000
)

np.nanmean(df_comp.query("label == 'pkg_dk' and type == 'corr_pr' and peak_present == 0")["value"].values)
np.nanstd(df_comp.query("label == 'pkg_dk' and type == 'corr_pr' and peak_present == 0")["value"].values)

nm_stats.permutationTest(
    df_comp.query("label == 'pkg_dk' and type=='corr_pr' and peak_present == 0")["value"].values,
    np.zeros(df_comp.query("label == 'pkg_dk' and type=='corr_pr' and peak_present == 0")["value"].values.shape[0]),
    False, None, 5000
)

# 
df_comp_withnight.query("label == 'pkg_bk' and type == 'corr_pr'")["value"].values.mean()
df_comp_withnight.query("label == 'pkg_bk' and type == 'corr_pr'")["value"].values.std()

nm_stats.permutationTest_relative(
    df_comp.query("label == 'pkg_bk' and type=='corr_pr'")["value"].values,
    df_comp_withnight.query("label == 'pkg_bk' and type=='corr_pr'")["value"].values,
    False, None, 5000
)

df_comp_withnight.query("label == 'pkg_tremor' and type == 'corr_pr'")["value"].values.mean()
df_comp_withnight.query("label == 'pkg_tremor' and type == 'corr_pr'")["value"].values.std()

nm_stats.permutationTest_relative(
    df_comp.query("label == 'pkg_tremor' and type=='corr_pr'")["value"].values,
    df_comp_withnight.query("label == 'pkg_tremor' and type=='corr_pr'")["value"].values,
    False, None, 5000
)

df_comp_withnight.query("label == 'pkg_dk' and type == 'corr_pr'")["value"].values.mean()
df_comp_withnight.query("label == 'pkg_dk' and type == 'corr_pr'")["value"].values.std()

nm_stats.permutationTest_relative(
    df_comp.query("label == 'pkg_dk' and type=='corr_pr'")["value"].values,
    df_comp_withnight.query("label == 'pkg_dk' and type=='corr_pr'")["value"].values,
    False, None, 5000
)

26/30

df_peaks = pd.DataFrame.from_dict(ind_peaks_long, orient="index", columns=["peak"])
df_peaks["sub"] = df_peaks.index
df_peaks["loc"] = "STN"
df_peaks.loc[df_peaks["sub"].isin(subs_GP), "loc"] = "GP"

df_peaks["peak"].mean() # 22.22
df_peaks["peak"].std()  # 5.77

df_peaks.query("loc == 'STN'")["peak"].mean() # 22.14
df_peaks.query("loc == 'STN'")["peak"].std()  # 5.69

df_peaks.query("loc == 'GP'")["peak"].mean() # 22.5
df_peaks.query("loc == 'GP'")["peak"].std()  # 6.65

# patients_without_peak = ["rcs05r", "rcs07r", "rcs12l", "rcs18l", "rcs20l"] short
patients_without_peak = [ "rcs08l", "rcs10l", "rcs12r", "rcs18l"]
order_ = ["pkg_bk", "pkg_tremor", "pkg_dk"]

df_plt_ = df_comp.copy()

df_stat = df_plt_[~df_plt_["sub"].isin(patients_without_peak)]
df_plt_["loc"] = "STN"
df_plt_.loc[df_plt_["sub"].isin(subs_GP), "loc"] = "GP"
df_plt_.query("type == 'corr_pr' and peak_present == 0").groupby(["loc", "label"])["value"].mean()
df_plt_.query("type == 'corr_pr' and peak_present == 0").groupby(["loc", "label"])["value"].std()

# PKG bk

df_stat.query("label == 'pkg_bk' and type == 'corr_ind'")["value"].values.mean()
df_stat.query("label == 'pkg_bk' and type == 'corr_ind'")["value"].values.std()

nm_stats.permutationTest(
    df_stat.query("label == 'pkg_bk' and type == 'corr_ind'")["value"].values,
    np.zeros(df_stat.query("label == 'pkg_bk' and type == 'corr_ind'")["value"].values.shape[0]),
    False, None, 5000
)

# PKG tremor

df_stat.query("label == 'pkg_tremor' and type == 'corr_ind'")["value"].values.mean()
df_stat.query("label == 'pkg_tremor' and type == 'corr_ind'")["value"].values.std()

nm_stats.permutationTest(
    df_stat.query("label == 'pkg_tremor' and type == 'corr_ind'")["value"].values,
    np.zeros(df_stat.query("label == 'pkg_tremor' and type == 'corr_ind'")["value"].values.shape[0]),
    False, None, 5000
)

# PKG dk

df_stat.query("label == 'pkg_dk' and type == 'corr_ind'")["value"].values.mean()
df_stat.query("label == 'pkg_dk' and type == 'corr_ind'")["value"].values.std()

nm_stats.permutationTest(
    df_stat.query("label == 'pkg_dk' and type == 'corr_ind'")["value"].values,
    np.zeros(df_stat.query("label == 'pkg_dk' and type == 'corr_ind'")["value"].values.shape[0]),
    False, None, 5000
)
# ML performances
df_stat.query("label == 'pkg_bk' and type == 'corr_pr'")["value"].values.mean()
df_stat.query("label == 'pkg_bk' and type == 'corr_pr'")["value"].values.std()

nm_stats.permutationTest(
    df_stat.query("label == 'pkg_bk' and type == 'corr_pr'")["value"].values,
    np.zeros(df_stat.query("label == 'pkg_bk' and type == 'corr_pr'")["value"].values.shape[0]),
    False, None, 5000
)

# Tr
df_stat.query("label == 'pkg_tremor' and type == 'corr_pr'")["value"].values.mean()
df_stat.query("label == 'pkg_tremor' and type == 'corr_pr'")["value"].values.std()

nm_stats.permutationTest(
    df_stat.query("label == 'pkg_tremor' and type == 'corr_pr'")["value"].values,
    np.zeros(df_stat.query("label == 'pkg_tremor' and type == 'corr_pr'")["value"].values.shape[0]),
    False, None, 5000
)
# DK
df_stat.query("label == 'pkg_dk' and type == 'corr_pr'")["value"].values.mean()
df_stat.query("label == 'pkg_dk' and type == 'corr_pr'")["value"].values.std()

nm_stats.permutationTest(
    df_stat.query("label == 'pkg_bk' and type == 'corr_pr'")["value"].values,
    np.zeros(df_stat.query("label == 'pkg_bk' and type == 'corr_pr'")["value"].values.shape[0]),
    False, None, 5000
)

# now differences

(df_stat.query("label == 'pkg_bk' and type == 'corr_pr'")["value"].values -
 df_stat.query("label == 'pkg_bk' and type == 'corr_ind'")["value"].values).mean()

(df_stat.query("label == 'pkg_bk' and type == 'corr_pr'")["value"].values -
 df_stat.query("label == 'pkg_bk' and type == 'corr_ind'")["value"].values).std()

nm_stats.permutationTest_relative(
    df_stat.query("label == 'pkg_bk' and type == 'corr_pr'")["value"].values,
    df_stat.query("label == 'pkg_bk' and type == 'corr_ind'")["value"].values,
    False, None, 5000
)
# tr
(df_stat.query("label == 'pkg_tremor' and type == 'corr_pr'")["value"].values -
 df_stat.query("label == 'pkg_tremor' and type == 'corr_ind'")["value"].values).mean()

(df_stat.query("label == 'pkg_tremor' and type == 'corr_pr'")["value"].values -
 df_stat.query("label == 'pkg_tremor' and type == 'corr_ind'")["value"].values).std()

nm_stats.permutationTest_relative(
    df_stat.query("label == 'pkg_tremor' and type == 'corr_pr'")["value"].values,
    df_stat.query("label == 'pkg_tremor' and type == 'corr_ind'")["value"].values,
    False, None, 5000
)

# dk
(df_stat.query("label == 'pkg_dk' and type == 'corr_pr'")["value"].values -
 df_stat.query("label == 'pkg_dk' and type == 'corr_ind'")["value"].values*-1).mean()

(df_stat.query("label == 'pkg_dk' and type == 'corr_pr'")["value"].values -
 df_stat.query("label == 'pkg_dk' and type == 'corr_ind'")["value"].values*-1).std()

nm_stats.permutationTest_relative(
    df_stat.query("label == 'pkg_dk' and type == 'corr_pr'")["value"].values,
    df_stat.query("label == 'pkg_dk' and type == 'corr_ind'")["value"].values,
    False, None, 5000
)