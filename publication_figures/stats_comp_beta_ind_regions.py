import pandas as pd
import os
from py_neuromodulation import nm_stats

PATH_PER = r'/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/paper_per/without_night/ind_ch'

PATH_FIGURES = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf/figures_paper/figures_final"
df = pd.read_csv(os.path.join(PATH_PER, "df_per_ind_all_coords.csv"), index_col=0)

PATH_PER = "publication_figures"
df_comp = pd.read_csv(os.path.join(PATH_PER, "df_comp_beta_ml.csv"))

subs_GP = df.query("loc == 'GP' and classification == False")["sub"].unique()
subs_STN = df.query("loc == 'STN' and classification == False")["sub"].unique()

# beta correlations GP
df_comp.query("label == 'pkg_bk' and sub in @subs_GP and type == 'corr_ind' and sub != 'rcs10l'")["value"].mean()
df_comp.query("label == 'pkg_bk' and sub in @subs_GP and type == 'corr_ind' and sub != 'rcs10l'")["value"].std()
# 0.47 +- 0.18


# beta correlations STN
subs_no_peak_present_STN = ["rcs08l", "rcs12r", "rcs18l"]
df_comp.query("label == 'pkg_tremor' and sub in @subs_STN and type == 'corr_ind' and sub not in @subs_no_peak_present_STN")["value"].values.mean()
df_comp.query("label == 'pkg_tremor' and sub in @subs_STN and type == 'corr_ind' and sub not in @subs_no_peak_present_STN")["value"].values.std()
# 0.24 +- 0.41

# STN
subs_no_peak_present_STN = ["rcs08l", "rcs12r", "rcs18l"]
nm_stats.permutationTest_relative(
    df.query("label == 'pkg_bk' and classification == False and loc == 'STN' and sub not in @subs_no_peak_present_STN").groupby("sub")["per"].mean().values,
    df_comp.query("label == 'pkg_bk' and sub in @subs_STN and type == 'corr_ind' and sub not in @subs_no_peak_present_STN")["value"].values,
    False, None, 5000
)# bk: p=0.046

subs_no_peak_present_STN = ["rcs08l", "rcs12r", "rcs18l"]
nm_stats.permutationTest_relative(
    df.query("label == 'pkg_tremor' and classification == False and loc == 'STN' and sub not in @subs_no_peak_present_STN").groupby("sub")["per"].mean().values,
    df_comp.query("label == 'pkg_tremor' and sub in @subs_STN and type == 'corr_ind' and sub not in @subs_no_peak_present_STN")["value"].values,
    False, None, 5000
)# tremor: p=0.70


# first group across subjects (mean)  --> then std
# for some reason entry's are doubled here.. 
df.query("label == 'pkg_bk' and classification == False and loc == 'STN'").groupby("sub")["per"].mean().mean()
df.query("label == 'pkg_bk' and classification == False and loc == 'STN'").groupby("sub")["per"].mean().std()
# bk 0.47 +- 0.24
# tremor: 0.25 +- 0.31
df.query("label == 'pkg_tremor' and classification == False and loc == 'STN'").groupby("sub")["per"].mean().mean()
df.query("label == 'pkg_tremor' and classification == False and loc == 'STN'").groupby("sub")["per"].mean().std()

# GP:
df.query("label == 'pkg_bk' and classification == False and loc == 'GP'").groupby("sub")["per"].mean().mean()
df.query("label == 'pkg_bk' and classification == False and loc == 'GP'").groupby("sub")["per"].mean().std()
df.query("label == 'pkg_tremor' and classification == False and loc == 'GP'").groupby("sub")["per"].mean().mean()
df.query("label == 'pkg_tremor' and classification == False and loc == 'GP'").groupby("sub")["per"].mean().std()

# bk: 0.23 +- 0.36
# tremor: 0.36 +- 0.31

nm_stats.permutationTest_relative(
    df.query("label == 'pkg_bk' and classification == False and loc == 'GP' and sub != 'rcs10l'").groupby("sub")["per"].mean().values,
    df_comp.query("label == 'pkg_bk' and sub in @subs_GP and type == 'corr_ind' and sub != 'rcs10l'")["value"].values,
    False, None, 5000
)# bk: p=0.38

nm_stats.permutationTest_relative(
    df.query("label == 'pkg_tremor' and classification == False and loc == 'GP' and sub != 'rcs10l'").groupby("sub")["per"].mean().values,
    df_comp.query("label == 'pkg_tremor' and sub in @subs_GP and type == 'corr_ind' and sub != 'rcs10l'")["value"].values,
    False, None, 5000
)# tremor: p=0.80




nm_stats.permutationTest()