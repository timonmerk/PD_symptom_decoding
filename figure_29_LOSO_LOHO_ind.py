import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import pickle
import seaborn as sns
import os

PATH_PER = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/paper_per"
PATH_FIGURES = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf/figures_paper"

# read f"ind_subjects_cvall_all.pkl"

l = []
for CLASSIFICATION in [True, False]:
    for pkg_label in ["pkg_dk", "pkg_bk", "pkg_tremor"]:
        with open(f"{PATH_PER}/loso_per_CLASS_{CLASSIFICATION}_{pkg_label}.pkl", "rb") as f:
            d_out = pickle.load(f)
        for sub in d_out[CLASSIFICATION][pkg_label].keys():
            l.append({
                "sub": sub,
                "pkg_label": pkg_label,
                "CLASSIFICATION": CLASSIFICATION,
                "per": d_out[CLASSIFICATION][pkg_label][sub]["per"]
            })
df_loso = pd.DataFrame(l)
df_loso["cv"] = "LOSO"


# PATH_PRE = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/LOSO_ALL_LABELS_ALL_GROUPS.pkl'
# with open(PATH_PRE, "rb") as f:
#     d_out = pickle.load(f)

l = []
for CLASSIFICATION in [True, False]:
    for pkg_label in ["pkg_dk", "pkg_bk", "pkg_tremor"]:
        #LOHO_main_pkg_tremor_CLASS_False_loc_stn_nonorm_withpsd
        with open(f"{PATH_PER}/LOHO_main_{pkg_label}_CLASS_{CLASSIFICATION}_loc_ecog_stn_nonorm_withpsd.pkl", "rb") as f:
            d_out = pickle.load(f)
        for sub in d_out.keys():
            if CLASSIFICATION is True:
                per_ = "ba"
            else:
                per_ = "corr_coeff"
            l.append({
                "sub": sub,
                "pkg_label": pkg_label,
                "CLASSIFICATION": CLASSIFICATION,
                "per": d_out[sub][per_]
            })
df_loho = pd.DataFrame(l)
df_loho["cv"] = "LOHO"

# read individual performances
PATH_PER = r'/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/paper_per/ind_ch'
df_ind = pd.read_csv(os.path.join(PATH_PER, "df_per_ind_all_coords.csv"), index_col=0)
# rename 'CLASSIFICATION' to 'classification'
df_ind = df_ind.rename(columns={"classification": "CLASSIFICATION"})
df_ind = df_ind.rename(columns={"label": "pkg_label"})
df_ind = df_ind.groupby(["CLASSIFICATION", "pkg_label", "sub"]).max().reset_index()
df_ind["cv"] = "ind"

df = pd.concat([df_loso, df_loho, df_ind], axis=0)

def get_stats_df(x, y, hue, df):
    from py_neuromodulation import nm_stats
    df_stats = []
    for hue_ in df[x].unique():
        for x_ in df[hue].unique():
            x_stat = df.query(f"{x} == @hue_ and {hue} == @x_")[y]
            for y_ in df[hue].unique():
                if x_ == y_:
                    continue
                y_stat = df.query(f"{x} == @hue_ and {hue} == @y_")[y]
                gT, p = nm_stats.permutationTest(x_stat, y_stat, False, None, 5000)
                df_stats.append({
                    x: x_,
                    y: y_,
                    hue: hue_,
                    "gT": gT,
                    "p": p
                })
    df_stats = pd.DataFrame(df_stats)
    return df_stats

def write_mean_std(df):
    # round to 2 decimal places
    df_stats = df.groupby(["pkg_label", "CLASSIFICATION", "cv"]).agg({"per": ["mean", "std"]}).reset_index()
    df_stats.columns = ["pkg_label", "CLASSIFICATION", "cv", "mean", "std"]
    df_stats = df_stats.round(2)
    return df_stats
   # df_stats.to_csv(os.path.join(PATH_PER, "df_stats_per.csv"))

plt.figure(figsize=(7, 4))
plt.subplot(121)
sns.boxplot(x="pkg_label", y="per", hue="cv", data=df.query("CLASSIFICATION == False"), showfliers=False, showmeans=True, palette="viridis", boxprops=dict(alpha=0.5))
sns.swarmplot(x="pkg_label", y="per", hue="cv", data=df.query("CLASSIFICATION == False"), alpha=0.5, dodge=True, size=2.5, palette="viridis")
df_t_test_ = get_stats_df("pkg_label", "per", "cv", df.query("CLASSIFICATION == False"))
df_stats = write_mean_std(df)
df_stats.to_csv(os.path.join(PATH_PER, "class_loho_loso_ind_per_table.csv"))
df_t_test_.to_csv(os.path.join(PATH_PER, "class_loho_loso_ind_per_t_test.csv"))
plt.ylabel("Correlation coefficient")
#plt.tight_layout()
#plt.savefig(os.path.join(PATH_FIGURES, "figure_29_LOSO_LOHO_ind_CLASS_False.pdf"))
#plt.show(block=True)

#plt.figure(figsize=(10, 5), dpi=300)
plt.subplot(122)
sns.boxplot(x="pkg_label", y="per", hue="cv", data=df.query("CLASSIFICATION == True"), showfliers=False, showmeans=True, palette="viridis", boxprops=dict(alpha=0.5))
sns.swarmplot(x="pkg_label", y="per", hue="cv", data=df.query("CLASSIFICATION == True"), alpha=0.5, dodge=True, size=2.5, palette="viridis")
df_t_test_ = get_stats_df("pkg_label", "per", "cv", df.query("CLASSIFICATION == True"))
df_stats = get_stats_df("pkg_label", "per", "cv", df.query("CLASSIFICATION == True"))

plt.ylabel("Balanced accuracy")
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "figure_29_LOSO_LOHO_ind_CLASS_True.pdf"))
plt.show(block=True)

