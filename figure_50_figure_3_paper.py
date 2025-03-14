import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import pickle
import os
import seaborn as sns


def read_per(d_out, CLASSIFICATION, pkg_label):
    l = []
    #for CLASSIFICATION in d_out.keys():
    if CLASSIFICATION is True:
        per_ = "ba"
    else:
        per_ = "corr_coeff"
    #for pkg_label in d_out[CLASSIFICATION].keys():
    for sub in d_out[CLASSIFICATION][pkg_label]["ecog_stn"].keys():
        l.append({
            "sub": sub,
            "pkg_label": pkg_label,
            "CLASSIFICATION": CLASSIFICATION,
            "per": d_out[CLASSIFICATION][pkg_label]["ecog_stn"][sub][per_]
        })
    df_loso = pd.DataFrame(l)
    return df_loso

PATH_PER = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/paper_per"
PATH_FIGURES = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf/figures_paper"
l_ = []
missing_files_ = []
for pkg_label in ["pkg_dk", "pkg_bk", "pkg_tremor"]:
    for CLASS_ in [True, False]:
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
df_h = pd.concat(l_, axis=0)

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
df_h = pd.concat([df_h, df_houronly], axis=0)

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
df_n = pd.concat(l_, axis=0)
df_n.to_csv(os.path.join(PATH_PER, "df_n.csv"))

PATH_PER_IND = r'/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/paper_per/ind_ch'
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
df_loso = pd.DataFrame(l)
df_loso["cv"] = "LOSO"

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

df_cval = pd.concat([df_loso, df_loho, df_ind], axis=0)

def set_box_alpha(ax, alpha=0.5):
    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, alpha))

plt.figure(figsize=(14, 7))
plt.subplot(2, 4, 1)
ax = sns.boxplot(data=df_h.query("CLASSIFICATION == True"), x="pkg_label", y="per", hue="hour_feature", palette="viridis", showmeans=True, showfliers=False, order=["pkg_bk", "pkg_dk", "pkg_tremor"]); set_box_alpha(ax)
sns.swarmplot(data=df_h.query("CLASSIFICATION == True"), x="pkg_label", y="per", hue="hour_feature", dodge=True, palette="viridis", alpha=0.9, s=2, order=["pkg_bk", "pkg_dk", "pkg_tremor"])
#df_stats = get_stats_df("pkg_label", "per", "hour_feature", df_h.query("CLASSIFICATION == True"))
plt.ylabel("Balanced accuracy")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.subplot(2, 4, 1+4)
ax = sns.boxplot(data=df_h.query("CLASSIFICATION == False"), x="pkg_label", y="per", hue="hour_feature", palette="viridis", showmeans=True, showfliers=False, order=["pkg_bk", "pkg_dk", "pkg_tremor"]); set_box_alpha(ax)
sns.swarmplot(data=df_h.query("CLASSIFICATION == False"), x="pkg_label", y="per", hue="hour_feature", dodge=True, palette="viridis", alpha=0.9, s=2, order=["pkg_bk", "pkg_dk", "pkg_tremor"])
#df_stats = get_stats_df("pkg_label", "per", "hour_feature", df_h.query("CLASSIFICATION == False"))
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.ylabel("Correlation coefficient")
plt.tight_layout()
#plt.savefig(os.path.join(PATH_FIGURES, "figure_35_per_exlude_hour_feature.pdf"))
plt.subplot(2, 4, 2)
ax = sns.boxplot(data=df_n.query("CLASSIFICATION == True"), x="pkg_label", y="per", hue="include_night", palette="viridis", showmeans=True, showfliers=False, order=["pkg_bk", "pkg_dk", "pkg_tremor"]); set_box_alpha(ax)
sns.swarmplot(data=df_n.query("CLASSIFICATION == True"), x="pkg_label", y="per", hue="include_night", dodge=True, palette="viridis", alpha=0.9, s=2, order=["pkg_bk", "pkg_dk", "pkg_tremor"])
#df_stats = get_stats_df("pkg_label", "per", "include_night", df_n.query("CLASSIFICATION == True"))
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.ylabel("Balanced accuracy")
plt.subplot(2, 4, 2+4)
ax = sns.boxplot(data=df_n.query("CLASSIFICATION == False"), x="pkg_label", y="per", hue="include_night", palette="viridis", showmeans=True, showfliers=False, order=["pkg_bk", "pkg_dk", "pkg_tremor"]); set_box_alpha(ax)
sns.swarmplot(data=df_n.query("CLASSIFICATION == False"), x="pkg_label", y="per", hue="include_night", dodge=True, palette="viridis", alpha=0.9, s=2, order=["pkg_bk", "pkg_dk", "pkg_tremor"])
#df_stats = get_stats_df("pkg_label", "per", "include_night", df_n.query("CLASSIFICATION == False"))
plt.ylabel("Correlation coefficient")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.subplot(2, 4, 3)
ax = sns.boxplot(x="label", y="per", hue="loc", hue_order=regions_hue_order, order=["pkg_bk", "pkg_dk", "pkg_tremor"],
            data=df_regions.query("classification == True"), palette="viridis", showfliers=False,
            showmeans=True); set_box_alpha(ax, 0.5)
sns.swarmplot(x="label", y="per", hue="loc",
              hue_order=regions_hue_order, data=df_regions.query("classification == True"), order=["pkg_bk", "pkg_dk", "pkg_tremor"],
              palette="viridis", alpha=0.5, dodge=True, size=2.5)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.title("Region-wise performances")

plt.subplot(2, 4, 3+4)
ax = sns.boxplot(x="label", y="per", hue="loc", hue_order=regions_hue_order, order=["pkg_bk", "pkg_dk", "pkg_tremor"],
            data=df_regions.query("classification == False"), palette="viridis", showfliers=False,
            showmeans=True); set_box_alpha(ax, 0.5)
sns.swarmplot(x="label", y="per", hue="loc",
              hue_order=regions_hue_order, data=df_regions.query("classification == False"),order=["pkg_bk", "pkg_dk", "pkg_tremor"],
              palette="viridis", alpha=0.5, dodge=True, size=2.5)
plt.ylabel("Correlation coefficient")
plt.title("Region-wise performances")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)

plt.subplot(2, 4, 4)
sns.boxplot(x="pkg_label", y="per", hue="cv", data=df_cval.query("CLASSIFICATION == True"), showfliers=False, showmeans=True, palette="viridis", boxprops=dict(alpha=0.5), order=["pkg_bk", "pkg_dk", "pkg_tremor"])
sns.swarmplot(x="pkg_label", y="per", hue="cv", data=df_cval.query("CLASSIFICATION == True"), alpha=0.5, dodge=True, size=2.5, palette="viridis", order=["pkg_bk", "pkg_dk", "pkg_tremor"])
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.ylabel("Balanced accuracy")

plt.subplot(2, 4, 4+4)
sns.boxplot(x="pkg_label", y="per", hue="cv", data=df_cval.query("CLASSIFICATION == False"), showfliers=False, showmeans=True, palette="viridis", boxprops=dict(alpha=0.5), order=["pkg_bk", "pkg_dk", "pkg_tremor"])
sns.swarmplot(x="pkg_label", y="per", hue="cv", data=df_cval.query("CLASSIFICATION == False"), alpha=0.5, dodge=True, size=2.5, palette="viridis", order=["pkg_bk", "pkg_dk", "pkg_tremor"])
plt.ylabel("Correlation coefficient")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)


plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "figure_3_comb.pdf"))
plt.show(block=True)
