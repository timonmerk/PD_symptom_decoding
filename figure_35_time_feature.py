import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import pickle
import os
import seaborn as sns


def read_per(d_out):
    l = []
    for CLASSIFICATION in d_out.keys():
        if CLASSIFICATION is True:
            per_ = "ba"
        else:
            per_ = "corr_coeff"
        for pkg_label in d_out[CLASSIFICATION].keys():
            for sub in d_out[CLASSIFICATION][pkg_label]["ecog_stn"].keys():
                l.append({
                    "sub": sub,
                    "pkg_label": pkg_label,
                    "CLASSIFICATION": CLASSIFICATION,
                    "per": d_out[CLASSIFICATION][pkg_label]["ecog_stn"][sub][per_]
                })
    df_loso = pd.DataFrame(l)
    return df_loso

PATH_PER = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per"
PATH_FIGURES = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf"
l_ = []
for exclude_hour in [True, False]:
    file = f"LOHO_ALL_LABELS_ALL_GROUPS_exludehour_{exclude_hour}.pkl"
    with open(os.path.join(PATH_PER, file), "rb") as f:
        d_out = pickle.load(f)
        df_ = read_per(d_out)
        df_["hour_feature"] = not exclude_hour
        l_.append(df_)
df_h = pd.concat(l_, axis=0)

file = f"LOHO_ALL_LABELS_ALL_GROUPS_HOUR_ONLY.pkl"
with open(os.path.join(PATH_PER, file), "rb") as f:
    d_out = pickle.load(f)
    df_ = read_per(d_out)
    df_["hour_feature"] = "hour_only"

df_h = pd.concat([df_h, df_], axis=0)



l_ = []
for exclude_night in [True, False]:
    file = f"LOHO_ALL_LABELS_ALL_GROUPS_exludenight_{exclude_night}.pkl"
    with open(os.path.join(PATH_PER, file), "rb") as f:
        d_out = pickle.load(f)
        df_ = read_per(d_out)
        df_["include_night"] = not exclude_night
        l_.append(df_)
df_n = pd.concat(l_, axis=0)

def set_box_alpha(ax, alpha=0.5):
    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, alpha))

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
                gT, p = nm_stats.permutationTest_relative(np.array(x_stat), np.array(y_stat), False, None, 5000)
                df_stats.append({
                    "x": x_,
                    "y": y_,
                    "mean diff" : x_stat.mean() - y_stat.mean(),
                    "std diff": np.std(x_stat - y_stat),
                     x: hue_,
                    "gT": gT,
                    "p": p
                })
    df_stats = pd.DataFrame(df_stats)
    return df_stats

print(df_h.query("CLASSIFICATION == False and pkg_label == 'pkg_bk' and hour_feature == True")["per"].mean())
print(df_h.query("CLASSIFICATION == False and pkg_label == 'pkg_bk' and hour_feature == True")["per"].std())

print(df_h.query("CLASSIFICATION == False and pkg_label == 'pkg_bk' and hour_feature == False")["per"].mean())
print(df_h.query("CLASSIFICATION == False and pkg_label == 'pkg_bk' and hour_feature == False")["per"].std())

print(df_h.query("CLASSIFICATION == True and pkg_label == 'pkg_dk' and hour_feature == True")["per"].mean())
print(df_h.query("CLASSIFICATION == True and pkg_label == 'pkg_dk' and hour_feature == True")["per"].std())

print(df_h.query("CLASSIFICATION == True and pkg_label == 'pkg_dk' and hour_feature == False")["per"].mean())
print(df_h.query("CLASSIFICATION == True and pkg_label == 'pkg_dk' and hour_feature == False")["per"].std())

print(df_h.query("CLASSIFICATION == True and pkg_label == 'pkg_tremor' and hour_feature == True")["per"].mean())
print(df_h.query("CLASSIFICATION == True and pkg_label == 'pkg_tremor' and hour_feature == True")["per"].std())

print(df_h.query("CLASSIFICATION == True and pkg_label == 'pkg_tremor' and hour_feature == False")["per"].mean())
print(df_h.query("CLASSIFICATION == True and pkg_label == 'pkg_tremor' and hour_feature == False")["per"].std())


## night-time

print(df_n.query("CLASSIFICATION == False and pkg_label == 'pkg_bk' and include_night == True")["per"].mean())
print(df_n.query("CLASSIFICATION == False and pkg_label == 'pkg_bk' and include_night == True")["per"].std())

print(df_n.query("CLASSIFICATION == False and pkg_label == 'pkg_bk' and include_night == False")["per"].mean())
print(df_n.query("CLASSIFICATION == False and pkg_label == 'pkg_bk' and include_night == False")["per"].std())

print(df_n.query("CLASSIFICATION == True and pkg_label == 'pkg_dk' and include_night == True")["per"].mean())
print(df_n.query("CLASSIFICATION == True and pkg_label == 'pkg_dk' and include_night == True")["per"].std())

print(df_n.query("CLASSIFICATION == True and pkg_label == 'pkg_dk' and include_night == False")["per"].mean())
print(df_n.query("CLASSIFICATION == True and pkg_label == 'pkg_dk' and include_night == False")["per"].std())
print(df_n.query("CLASSIFICATION == True and pkg_label == 'pkg_tremor' and include_night == True")["per"].mean())
print(df_n.query("CLASSIFICATION == True and pkg_label == 'pkg_tremor' and include_night == True")["per"].std())

print(df_n.query("CLASSIFICATION == True and pkg_label == 'pkg_tremor' and include_night == False")["per"].mean())
print(df_n.query("CLASSIFICATION == True and pkg_label == 'pkg_tremor' and include_night == False")["per"].std())

plt.figure(figsize=(7, 7), dpi=300)
plt.subplot(2, 2, 1)
ax = sns.boxplot(data=df_h.query("CLASSIFICATION == True"), x="pkg_label", y="per", hue="hour_feature", palette="viridis", showmeans=True, showfliers=False, order=["pkg_bk", "pkg_dk", "pkg_tremor"]); set_box_alpha(ax)
sns.swarmplot(data=df_h.query("CLASSIFICATION == True"), x="pkg_label", y="per", hue="hour_feature", dodge=True, palette="viridis", alpha=0.9, s=2, order=["pkg_bk", "pkg_dk", "pkg_tremor"])
#df_stats = get_stats_df("pkg_label", "per", "hour_feature", df_h.query("CLASSIFICATION == True"))
plt.ylabel("Balanced accuracy")
plt.subplot(2, 2, 2)
ax = sns.boxplot(data=df_h.query("CLASSIFICATION == False"), x="pkg_label", y="per", hue="hour_feature", palette="viridis", showmeans=True, showfliers=False, order=["pkg_bk", "pkg_dk", "pkg_tremor"]); set_box_alpha(ax)
sns.swarmplot(data=df_h.query("CLASSIFICATION == False"), x="pkg_label", y="per", hue="hour_feature", dodge=True, palette="viridis", alpha=0.9, s=2, order=["pkg_bk", "pkg_dk", "pkg_tremor"])
#df_stats = get_stats_df("pkg_label", "per", "hour_feature", df_h.query("CLASSIFICATION == False"))

plt.ylabel("Correlation coefficient")
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "figure_35_per_exlude_hour_feature.pdf"))
plt.subplot(2, 2, 3)
ax = sns.boxplot(data=df_n.query("CLASSIFICATION == True"), x="pkg_label", y="per", hue="include_night", palette="viridis", showmeans=True, showfliers=False, order=["pkg_bk", "pkg_dk", "pkg_tremor"]); set_box_alpha(ax)
sns.swarmplot(data=df_n.query("CLASSIFICATION == True"), x="pkg_label", y="per", hue="include_night", dodge=True, palette="viridis", alpha=0.9, s=2, order=["pkg_bk", "pkg_dk", "pkg_tremor"])
#df_stats = get_stats_df("pkg_label", "per", "include_night", df_n.query("CLASSIFICATION == True"))

plt.ylabel("Balanced accuracy")
plt.subplot(2, 2, 4)
ax = sns.boxplot(data=df_n.query("CLASSIFICATION == False"), x="pkg_label", y="per", hue="include_night", palette="viridis", showmeans=True, showfliers=False, order=["pkg_bk", "pkg_dk", "pkg_tremor"]); set_box_alpha(ax)
sns.swarmplot(data=df_n.query("CLASSIFICATION == False"), x="pkg_label", y="per", hue="include_night", dodge=True, palette="viridis", alpha=0.9, s=2, order=["pkg_bk", "pkg_dk", "pkg_tremor"])
#df_stats = get_stats_df("pkg_label", "per", "include_night", df_n.query("CLASSIFICATION == False"))

plt.ylabel("Correlation coefficient")
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "figure_35_per_exclude_analysis_with_houronly.pdf"))
plt.show(block=True)


#####
plt.figure(figsize=(10, 7), dpi=300)
plt.subplot(1, 2, 1)
ax = sns.boxplot(data=df_h.query("CLASSIFICATION == True"),
                 x="pkg_label", y="per", hue="hour_feature", palette="viridis", showmeans=True, showfliers=False,
                 order=["pkg_bk", "pkg_dk", "pkg_tremor"]); set_box_alpha(ax)
sns.swarmplot(data=df_h.query("CLASSIFICATION == True"), x="pkg_label", y="per", hue="hour_feature", dodge=True,
              palette="viridis", alpha=0.9, s=2, order=["pkg_bk", "pkg_dk", "pkg_tremor"])
plt.ylabel("Balanced accuracy")
plt.subplot(1, 2, 2)
ax = sns.boxplot(data=df_h.query("CLASSIFICATION == False"), x="pkg_label", y="per",order=["pkg_bk", "pkg_dk", "pkg_tremor"],
                 hue="hour_feature", palette="viridis", showmeans=True, showfliers=False); set_box_alpha(ax)
sns.swarmplot(data=df_h.query("CLASSIFICATION == False"), x="pkg_label", y="per", hue="hour_feature",
              dodge=True, palette="viridis", alpha=0.9, s=2, order=["pkg_bk", "pkg_dk", "pkg_tremor"])
plt.ylabel("Correlation coefficient")
plt.show(block=True)