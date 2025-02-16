import pandas as pd

import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from py_neuromodulation import nm_stats
from scipy import stats
import seaborn as sns
import pickle
import yaml

PATH_PER = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per'
OUT_FILE = "LOHO_ALL_LABELS_ALL_GROUPS_normed_480.pkl"
PATH_FIGURES = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf'
#OUT_FILE = "LOHO_ALL_LABELS_ALL_GROUPS_nonorm.pkl"
PATH_READ = os.path.join(PATH_PER, OUT_FILE)

with open(PATH_READ, "rb") as f:
    d_out = pickle.load(f)

data = []
for CLASSIFICATION in d_out.keys():
    if CLASSIFICATION:
        per_ = "ba"
    else:
        per_ = "corr_coeff"
    for pkg_decode_label in d_out[CLASSIFICATION].keys():
        for loc in d_out[CLASSIFICATION][pkg_decode_label].keys():
            for sub in d_out[CLASSIFICATION][pkg_decode_label][loc].keys():
                data.append({
                    "CLASSIFICATION": CLASSIFICATION,
                    "per": d_out[CLASSIFICATION][pkg_decode_label][loc][sub][per_],
                    "sub": sub,
                    "pkg_decode_label": pkg_decode_label,
                    "loc": loc
                })

df = pd.DataFrame(data)

def set_box_alpha(ax, alpha=0.5):
    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, alpha))

plt.figure(figsize=(3, 8.2))
idx_plt = 0
x_order = ["ecog", "stn", "ecog_stn"]
for label in ["pkg_bk", "pkg_dk", "pkg_tremor"]:
    for CLASSIFICATION in [True, False]:
        idx_plt += 1
        plt.subplot(3, 2, idx_plt)
        ax = sns.boxplot(data=df.query("CLASSIFICATION == @CLASSIFICATION and pkg_decode_label == @label"), x="loc", y="per", palette="viridis", showfliers=False, showmeans=True, order=x_order); set_box_alpha(ax)
        sns.swarmplot(data=df.query("CLASSIFICATION == @CLASSIFICATION and pkg_decode_label == @label"), x="loc", y="per", dodge=False, color=".25", palette="viridis", alpha=0.5, s=3, order=x_order)
        plt.ylabel("Balanced Accuracy" if CLASSIFICATION else "Correlation Coefficient")
        plt.title(f"{label} {'CLASS' if CLASSIFICATION else 'REG'}")
        plt.gca().spines['right'].set_visible(False)
        plt.gca().spines['top'].set_visible(False)
        plt.xticks([1, 2, 3], ["ECoG", "SC", "ECoG+SC"], rotation=90)
        plt.xlabel("")

plt.savefig(os.path.join(PATH_FIGURES, "figure_34_per_groups.pdf"))
plt.tight_layout()
plt.show(block=True)

# write for each condition the mean and std
print(df.query("CLASSIFICATION == True").groupby(["pkg_decode_label", "loc"])["per"].mean().round(3))
print(df.query("CLASSIFICATION == True").groupby(["pkg_decode_label", "loc"])["per"].std().round(3))

loc_pairs = [("ecog", "stn"), ("ecog", "ecog_stn"), ("stn", "ecog_stn")]
l_p = []
l_per = []
for label in ["pkg_dk", "pkg_tremor", "pkg_bk"]:
    print(f"Label: {label}")
    for CLASSIFICATION in [True, False]:
        print(f"Classification: {CLASSIFICATION}")
        for loc in ["ecog", "stn", "ecog_stn"]:
            mean_ = df.query("CLASSIFICATION == @CLASSIFICATION and loc == @loc and pkg_decode_label == @label")["per"].mean()
            std_ = df.query("CLASSIFICATION == @CLASSIFICATION and loc == @loc and pkg_decode_label == @label")["per"].std()
            l_per.append({
                "CLASSIFICATION": CLASSIFICATION,
                "label": label,
                "loc": loc,
                "per" : f"{mean_:.3f} ± {std_:.3f}"
            }
            )

        for loc1, loc2 in loc_pairs:
            print(f"loc1: {loc1}, loc2: {loc2}")
            gt, p = nm_stats.permutationTest_relative(
                df.query("CLASSIFICATION == @CLASSIFICATION and loc == @loc1 and pkg_decode_label == @label")["per"],
                df.query("CLASSIFICATION == @CLASSIFICATION and loc == @loc2 and pkg_decode_label == @label")["per"],
                False, None, 5000
            )
            l_p.append({
                "CLASSIFICATION": CLASSIFICATION,
                "label": label,
                "loc1": loc1,
                "loc2": loc2,
                #"gt": gt,
                "p": p
            })
df_p = pd.DataFrame(l_p)
df_p.to_csv(os.path.join(PATH_PER, "p_values_LOHO.csv"), index=False)
print(df_p)

df_per = pd.DataFrame(l_per)
print(df_per)
df_per.to_csv(os.path.join(PATH_PER, "performance_values_LOHO.csv"), index=False)

print()

df_s = pd.read_csv("ClinicalScoresTable.csv")

with open("ucsf_config.yaml", "r") as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
    PATH_FEATURES = os.path.join(config["path_base"], config["features"])
    PATH_FIGURES = os.path.join(config["path_base"], config["figures"])

PATH_PKG = os.path.join(config["path_base"], "pkg_data")

PATH_OUT = os.path.join(PATH_FEATURES, "merged")
subs = np.sort([f[:6] for f in os.listdir(PATH_OUT) if "rcs" in f])

# idea here: check only if tremor is consistent
LIMIT_TO_DAYTIME = False

d_out = []
for sub in subs:
    df_pkg = pd.read_csv(os.path.join(PATH_PKG, f"{sub}_pkg.csv"))
    df_pkg.index = pd.to_datetime(df_pkg.pkg_dt)
    df_pkg["h"] = df_pkg.index.hour
    if LIMIT_TO_DAYTIME:
        df_pkg = df_pkg[(df_pkg.h >= 12) & (df_pkg.h <= 18)]
    if sub[-1] == "l":
        UE = "LUE"
        LE = "LLE"
        postural = "postural L"
        kinetic = "kinetic tremor L"
        updrs_tremor = "UPDRS tremor L"
    else:
        UE = "RUE"
        LE = "RLE"
        postural = "postural R"
        kinetic = "kinetic tremor R"
        updrs_tremor = "UPDRS tremor R"
    tremor_constancy = "constancy of tremor"

    UE = df_s[df_s["Study id"] == sub[:-1]][UE].iloc[0]
    LE = df_s[df_s["Study id"] == sub[:-1]][LE].iloc[0]

    # clip pkg_bk to 80
    df_pkg["pkg_bk"] = np.clip(df_pkg["pkg_bk"], 0, 300)
    
    d_out.append({
        "sub": sub[:-1],
        "UE": UE,
        "LE": LE,
        "postural" : df_s[df_s["Study id"] == sub[:-1]][postural].iloc[0],
        "kinetic" : df_s[df_s["Study id"] == sub[:-1]][kinetic].iloc[0],
        "updrs_tremor" : df_s[df_s["Study id"] == sub[:-1]][updrs_tremor].iloc[0],
        "tremor_constancy" : df_s[df_s["Study id"] == sub[:-1]][tremor_constancy].iloc[0],
        "pkg_tremor_mean": df_pkg["pkg_tremor"].mean(),
        "pkg_tremor_max": df_pkg["pkg_tremor"].max(),
        "pkg_tremor_75": np.quantile(df_pkg["pkg_tremor"].dropna(), 0.75),
        "pkg_dk_mean": df_pkg["pkg_dk"].mean(),
        "pkg_dk_max": df_pkg["pkg_dk"].max(),
        "pkg_dk_75": np.quantile(df_pkg["pkg_dk"].dropna(), 0.75),
        "pkg_bk_mean": df_pkg["pkg_bk"].mean(),
        "pkg_bk_median": df_pkg["pkg_bk"].median(),
        "pkg_bk_max": df_pkg["pkg_bk"].max(),
        "pkg_bk_75": np.quantile(df_pkg["pkg_bk"].dropna(), 0.75),
        "UPDRS (Off)" : df_s[df_s["Study id"] == sub[:-1]]["UPDRS (Off)"].iloc[0],
        "UPDRS (Off-On)" : df_s[df_s["Study id"] == sub[:-1]]["UPDRS (OFF-ON)"].iloc[0],
        "UPDRS IV" : df_s[df_s["Study id"] == sub[:-1]]["UPDRS IV"].iloc[0],
        "hem" : sub[-1]
    })

df_updrs = pd.DataFrame(d_out)

# merge the dataframes
df_per = df.copy()
df_per_ = df_per.query("CLASSIFICATION == False and loc == 'ecog_stn'").groupby(["sub", "pkg_decode_label"])["per"].mean().reset_index()
df_per_["sub_"] = df_per_["sub"].str[:-1]
df_per_ = df_per_.groupby(["sub_", "pkg_decode_label"])["per"].mean().reset_index()
#print(df_per_.head())
df_per_ = df_per_.rename(columns={"sub_": "sub"})
df_joint = df_per_.merge(df_updrs.groupby("sub")[["UPDRS IV", "UPDRS (Off)", "updrs_tremor"]].mean(), on="sub")
print(df_joint.head())

plt.figure()
plt.subplot(1, 3, 1)
sns.regplot(data=df_joint.query("pkg_decode_label == 'pkg_bk'"), x="UPDRS (Off)", y="per", scatter_kws={"alpha": 0.5})
rho, _ = stats.spearmanr(df_joint.query("pkg_decode_label == 'pkg_bk'")["UPDRS (Off)"], df_joint.query("pkg_decode_label == 'pkg_bk'")["per"])
_, p = nm_stats.permutationTestSpearmansRho(df_joint.query("pkg_decode_label == 'pkg_bk'")["UPDRS (Off)"], df_joint.query("pkg_decode_label == 'pkg_bk'")["per"], False, None, 5000)
plt.title(f"Bradykinesia \nrho={rho:.2f}, p={p:.3f}")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
#plt.ylabel("Balanced Accuracy")
plt.ylabel("Correlation Coefficient")
plt.subplot(1, 3, 2)
sns.regplot(data=df_joint.query("pkg_decode_label == 'pkg_dk'"), x="UPDRS IV", y="per", scatter_kws={"alpha": 0.5})
#plt.ylabel("Balanced Accuracy")
plt.ylabel("Correlation Coefficient")
rho, _ = stats.spearmanr(df_joint.query("pkg_decode_label == 'pkg_dk'")["UPDRS IV"], df_joint.query("pkg_decode_label == 'pkg_dk'")["per"])
_, p = nm_stats.permutationTestSpearmansRho(df_joint.query("pkg_decode_label == 'pkg_dk'")["UPDRS IV"], df_joint.query("pkg_decode_label == 'pkg_dk'")["per"], False, None, 5000)
plt.title(f"Dyskinesia \nrho={rho:.2f}, p={p:.3f}")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.subplot(1, 3, 3)
sns.regplot(data=df_joint.query("pkg_decode_label == 'pkg_tremor'"), x="updrs_tremor", y="per", scatter_kws={"alpha": 0.5})
#plt.ylabel("Balanced Accuracy")
plt.ylabel("Correlation Coefficient")
rho, _ = stats.spearmanr(df_joint.query("pkg_decode_label == 'pkg_tremor'")["updrs_tremor"], df_joint.query("pkg_decode_label == 'pkg_tremor'")["per"])
_, p = nm_stats.permutationTestSpearmansRho(df_joint.query("pkg_decode_label == 'pkg_tremor'")["updrs_tremor"], df_joint.query("pkg_decode_label == 'pkg_tremor'")["per"], False, None, 5000)
plt.title(f"Tremor \nrho={rho:.2f}, p={p:.3f}")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "figure_34_updrs_per_corr_corrcoeff.pdf"))
plt.show(block=True)