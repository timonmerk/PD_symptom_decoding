import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import pickle
import os
import seaborn as sns
from sklearn import metrics

def get_stats_df(x, y, hue, df):
    from py_neuromodulation import nm_stats
    df_stats = []
    for hue_ in df[x].unique():
        for x_ in df[hue].unique():
            x_stat = df.query(f"{x} == @hue_ and {hue} == @x_")[y].values
            for y_ in df[hue].unique():
                if x_ == y_:
                    continue
                y_stat = df.query(f"{x} == @hue_ and {hue} == @y_")[y].values
                nan_entries = np.isnan(x_stat) | np.isnan(y_stat)
                x_stat_ = x_stat[~nan_entries]
                y_stat_ = y_stat[~nan_entries]
                gT, p = nm_stats.permutationTest_relative(np.array(x_stat_), np.array(y_stat_), False, None, 5000)
                df_stats.append({
                    "x": x_,
                    "y": y_,
                    "mean diff" : x_stat_.mean() - y_stat_.mean(),
                    "std diff": np.std(x_stat_ - y_stat_),
                     x: hue_,
                    "gT": gT,
                    "p": p
                })
    df_stats = pd.DataFrame(df_stats)
    return df_stats



def read_per(d_out, CLASSIFICATION, pkg_label):
    l = []
    #for CLASSIFICATION in d_out.keys():
    if CLASSIFICATION is True:
        per_ = "ba"
    else:
        per_ = "corr_coeff"
    #for pkg_label in d_out[CLASSIFICATION].keys():
    for sub in d_out[CLASSIFICATION][pkg_label]["ecog_stn"].keys():
        hours_ = pd.to_datetime(d_out[CLASSIFICATION][pkg_label]["ecog_stn"][sub]["time"]).hour
        idx_sel = np.where(np.logical_and(hours_ >8, hours_ < 20))[0]
        pr = d_out[CLASSIFICATION][pkg_label]["ecog_stn"][sub]["pr"][idx_sel]
        true_ = d_out[CLASSIFICATION][pkg_label]["ecog_stn"][sub]["y_"][idx_sel]
        if CLASSIFICATION is True:
            per = metrics.balanced_accuracy_score(true_, pr)
        else:
            per = np.corrcoef(pr, true_)[0, 1]
        l.append({
            "sub": sub,
            "pkg_label": pkg_label,
            "CLASSIFICATION": CLASSIFICATION,
            "per": per, #d_out[CLASSIFICATION][pkg_label]["ecog_stn"][sub][per_]
        })
    df_loso = pd.DataFrame(l)
    return df_loso

PATH_PER = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/paper_per/without_night"
PATH_FIGURES = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf/figures_paper/figures_final"
l_ = []
missing_files_ = []
for pkg_label in ["pkg_bk", "pkg_dk", "pkg_tremor"]:
    for CLASS_ in [False, True]:
        for exclude_hour in [True, False]:
            file = f"LOHO_exludehour_nonorm_{exclude_hour}_CLASS_{CLASS_}_label_{pkg_label}_withpsd.pkl"
            #if os.path.exists(os.path.join(PATH_PER, file)) is False:
            #    missing_files_.append(file)
            #    continue
            with open(os.path.join(PATH_PER, file), "rb") as f:
                d_out = pickle.load(f)
                df_ = read_per(d_out, CLASS_, pkg_label)
                df_["hour_feature"] = not exclude_hour
            l_.append(df_)
df_h = pd.concat(l_, axis=0).reset_index(drop=True)

file = f"LOHO_ALL_LABELS_ALL_GROUPS_HOUR_ONLY.pkl"
l_ = []
for pkg_label in ["pkg_dk", "pkg_bk", "pkg_tremor"]:
    for CLASS_ in [True, False]:
        file = f"HOUR_ONLY_nonorm_CLASS_{CLASS_}_label_{pkg_label}_withpsd.pkl"
        with open(os.path.join(PATH_PER, file), "rb") as f:
            d_out = pickle.load(f)
            df_ = read_per(d_out, CLASS_, pkg_label)
            df_["hour_feature"] = "hour_only"
        l_.append(df_)
df_houronly = pd.concat(l_, axis=0)
df_h = pd.concat([df_h, df_houronly], axis=0).reset_index(drop=True)

l_ = []
for pkg_label in ["pkg_dk", "pkg_bk", "pkg_tremor"]:
    for CLASS_ in [True, False]:
        for exclude_night in [True, False]:
            #file = f"LOHO_ALL_LABELS_ALL_GROUPS_exludenight_{exclude_night}.pkl"
            file = f"LOHO_exludenight_nonorm_{exclude_night}_CLASS_{CLASS_}_label_{pkg_label}_withpsd.pkl"
            with open(os.path.join(PATH_PER, file), "rb") as f:
                d_out = pickle.load(f)
                df_ = read_per(d_out, CLASS_, pkg_label)
                df_["include_night"] = not exclude_night
                l_.append(df_)
df_n = pd.concat(l_, axis=0).reset_index(drop=True)
df_n.to_csv(os.path.join(PATH_PER, "df_n.csv"))

PATH_PER_IND = r'/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/paper_per/without_night/ind_ch'
df_regions = pd.read_csv(os.path.join(PATH_PER_IND, "df_per_ind_all_coords.csv"), index_col=0)
locs_new = []
for idx, row in df_regions.iterrows():
    if row["loc"] == "ECOG":
        if row["ch_orig"] == "8-9" or row["ch_orig"] == "8-10":
            locs_new.append("SC")
        else:
            locs_new.append("MC")
    else:
        locs_new.append(row["loc"])
df_regions["loc"] = locs_new
regions_hue_order = ["STN", "GP", "SC", "MC"]

df_ind = pd.read_csv(os.path.join(PATH_PER_IND, "df_per_ind_all_coords.csv"), index_col=0)
# rename 'CLASSIFICATION' to 'classification'
df_ind = df_ind.rename(columns={"classification": "CLASSIFICATION"})
df_ind = df_ind.rename(columns={"label": "pkg_label"})
df_ind = df_ind.groupby(["CLASSIFICATION", "pkg_label", "sub"]).max().reset_index()
df_ind["cv"] = "ind"

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
df_loso = pd.DataFrame(l).reset_index(drop=True)
df_loso["cv"] = "LOSO"

l = []
for CLASSIFICATION in [True, False]:
    for pkg_label in ["pkg_dk", "pkg_bk", "pkg_tremor"]:
        #LOHO_main_pkg_tremor_CLASS_False_loc_stn_nonorm_withpsd
        with open(f"{PATH_PER}/LOHO_main_{pkg_label}_CLASS_{CLASSIFICATION}_loc_ecog_stn_withpsd.pkl", "rb") as f:
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
df_loho = pd.DataFrame(l).reset_index(drop=True)
df_loho["cv"] = "LOHO"

df_cval = pd.concat([df_loso, df_loho, df_ind], axis=0)

df_cval.query("CLASSIFICATION == False").groupby(["cv", "pkg_label"])["per"].mean()
df_stats = get_stats_df("pkg_label", "per", "cv", df_cval.query("CLASSIFICATION == False"))


def set_box_alpha(ax, alpha=0.5):
    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, alpha))

label_order = ["pkg_bk", "pkg_tremor", "pkg_dk"]
plt.figure(figsize=(14, 7))

plt.subplot(2, 4, 1)
ax = sns.boxplot(data=df_h.query("CLASSIFICATION == False"), x="pkg_label", y="per", hue="hour_feature", palette="viridis", showmeans=True, showfliers=False, order=label_order); set_box_alpha(ax)
sns.swarmplot(data=df_h.query("CLASSIFICATION == False"), x="pkg_label", y="per", hue="hour_feature", dodge=True, palette="viridis", alpha=0.9, s=2, order=label_order)

# test here the individual stats
#df_stats = get_stats_df("pkg_label", "per", "hour_feature", df_h.query("CLASSIFICATION == False"))
# vals_ = df_h.query("CLASSIFICATION == False and pkg_label == 'pkg_tremor' and hour_feature == 'hour_only'")["per"].values
# vals_ = vals_[~np.isnan(vals_)]
# from py_neuromodulation import nm_stats
# nm_stats.permutationTest(
#     vals_, np.zeros(len(vals_)), False, None, 5000
# )

plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.subplot(2, 4, 1+4)
ax = sns.boxplot(data=df_h.query("CLASSIFICATION == True"), x="pkg_label", y="per", hue="hour_feature", palette="viridis", showmeans=True, showfliers=False, order=label_order); set_box_alpha(ax)
sns.swarmplot(data=df_h.query("CLASSIFICATION == True"), x="pkg_label", y="per", hue="hour_feature", dodge=True, palette="viridis", alpha=0.9, s=2, order=label_order)
#df_stats = get_stats_df("pkg_label", "per", "hour_feature", df_h.query("CLASSIFICATION == True"))
plt.ylabel("Balanced accuracy")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.ylabel("Correlation coefficient")
plt.tight_layout()
#plt.savefig(os.path.join(PATH_FIGURES, "figure_35_per_exlude_hour_feature.pdf"))

plt.subplot(2, 4, 2)
ax = sns.boxplot(data=df_n.query("CLASSIFICATION == False"), x="pkg_label", y="per", hue="include_night", palette="viridis", showmeans=True, showfliers=False, order=label_order); set_box_alpha(ax)
sns.swarmplot(data=df_n.query("CLASSIFICATION == False"), x="pkg_label", y="per", hue="include_night", dodge=True, palette="viridis", alpha=0.9, s=2, order=label_order, legend=False)
#df_stats = get_stats_df("pkg_label", "per", "include_night", df_n.query("CLASSIFICATION == False"))
#df_n.query("CLASSIFICATION == False and pkg_label == 'pkg_bk' and include_night == True")["per"].mean()
#df_n.query("CLASSIFICATION == False and pkg_label == 'pkg_bk' and include_night == True")["per"].std()
#df_n.query("CLASSIFICATION == False").groupby(["pkg_label", "include_night"])["per"].mean()
plt.ylabel("Correlation coefficient")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.subplot(2, 4, 2+4)
ax = sns.boxplot(data=df_n.query("CLASSIFICATION == True"), x="pkg_label", y="per", hue="include_night", palette="viridis", showmeans=True, showfliers=False, order=label_order); set_box_alpha(ax)
sns.swarmplot(data=df_n.query("CLASSIFICATION == True"), x="pkg_label", y="per", hue="include_night", dodge=True, palette="viridis", alpha=0.9, s=2, order=label_order)
df_stats = get_stats_df("pkg_label", "per", "include_night", df_n.query("CLASSIFICATION == True"))
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.ylabel("Balanced accuracy")

plt.subplot(2, 4, 3+4)
ax = sns.boxplot(x="label", y="per", hue="loc", hue_order=regions_hue_order, order=label_order,
            data=df_regions.query("classification == True"), palette="viridis", showfliers=False,
            showmeans=True); set_box_alpha(ax, 0.5)
sns.swarmplot(x="label", y="per", hue="loc",
              hue_order=regions_hue_order, data=df_regions.query("classification == True"), order=label_order,
              palette="viridis", alpha=0.5, dodge=True, size=2.5, legend=False)
df_regions.query("classification == False").query("label == 'pkg_bk'").groupby("loc")["per"].mean()
df_regions.query("classification == False").query("label == 'pkg_bk'").groupby("loc")["per"].std()
from py_neuromodulation import nm_stats
nm_stats.permutationTest(
    df_regions.query("classification == False").query("label == 'pkg_bk'").query("loc == 'STN'")["per"].values,
    df_regions.query("classification == False").query("label == 'pkg_bk'").query("loc == 'GP'")["per"].values,
    False, None, 5000
)
# Tremor
df_regions.query("classification == False").query("label == 'pkg_tremor'").groupby("loc")["per"].mean()
df_regions.query("classification == False").query("label == 'pkg_tremor'").groupby("loc")["per"].std()
from py_neuromodulation import nm_stats
nm_stats.permutationTest(
    df_regions.query("classification == False").query("label == 'pkg_tremor'").query("loc == 'STN'")["per"].values,
    df_regions.query("classification == False").query("label == 'pkg_tremor'").query("loc == 'GP'")["per"].values,
    False, None, 5000
)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.title("Region-wise performances")

plt.subplot(2, 4, 3)
ax = sns.boxplot(x="label", y="per", hue="loc", hue_order=regions_hue_order, order=label_order,
            data=df_regions.query("classification == False"), palette="viridis", showfliers=False,
            showmeans=True); set_box_alpha(ax, 0.5)
sns.swarmplot(x="label", y="per", hue="loc",
              hue_order=regions_hue_order, data=df_regions.query("classification == False"),order=label_order,
              palette="viridis", alpha=0.5, dodge=True, size=2.5, legend=False)
plt.ylabel("Correlation coefficient")
plt.title("Region-wise performances")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.subplot(2, 4, 4+4)
sns.boxplot(x="pkg_label", y="per", hue="cv", data=df_cval.query("CLASSIFICATION == True"), showfliers=False, showmeans=True, palette="viridis", boxprops=dict(alpha=0.5), order=label_order)
sns.swarmplot(x="pkg_label", y="per", hue="cv", data=df_cval.query("CLASSIFICATION == True"), alpha=0.5, dodge=True, size=2.5, palette="viridis", order=label_order, legend=False)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.ylabel("Balanced accuracy")

plt.subplot(2, 4, 4)
sns.boxplot(x="pkg_label", y="per", hue="cv", data=df_cval.query("CLASSIFICATION == False"), showfliers=False, showmeans=True, palette="viridis", boxprops=dict(alpha=0.5), order=label_order)
sns.swarmplot(x="pkg_label", y="per", hue="cv", data=df_cval.query("CLASSIFICATION == False"), alpha=0.5, dodge=True, size=2.5, palette="viridis", order=label_order, legend=False)
#df_cval.query("CLASSIFICATION == False").groupby(["cv", "pkg_label"])["per"].mean()
#df_stats = get_stats_df("pkg_label", "per", "cv", df_cval.query("CLASSIFICATION == False"))

plt.ylabel("Correlation coefficient")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)


plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "figure_3_comb.pdf"))
plt.show(block=True)
