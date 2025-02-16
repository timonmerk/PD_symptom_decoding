import pandas as pd    

import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


PATH_PER = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/out_dir"

PATH_FIGURES = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf"
df = pd.read_csv(os.path.join(PATH_PER, "df_per_ind_all_coords.csv"), index_col=0)
locs_new = []
for idx, row in df.iterrows():
    if row["loc"] == "ECOG":
        if row["ch_orig"] == "8-9" or row["ch_orig"] == "8-10":
            locs_new.append("SC")
        else:
            locs_new.append("MC")
    else:
        locs_new.append(row["loc"])
df["loc"] = locs_new
hue_order = ["STN", "GP", "SC", "MC"]

def set_box_alpha(ax, alpha=0.5):
    for patch in ax.patches:
        r, g, b, a = patch.get_facecolor()
        patch.set_facecolor((r, g, b, alpha))

from py_neuromodulation import nm_stats

# get Stats
for label_ in df["label"].unique():
    _, p = nm_stats.permutationTest(
        df.query("classification == True").query("label == @label_").query("loc == 'STN'")["per"].values,
        df.query("classification == True").query("label == @label_").query("loc == 'GP'")["per"].values,
        False, None, 5000
    )
    print(f"{label_} p: {p}")

plt.figure(figsize=(5, 5), dpi=300)
ax = sns.boxplot(x="label", y="per", hue="loc", hue_order=hue_order, order=["pkg_bk", "pkg_dk", "pkg_tremor"],
            data=df.query("classification == True"), palette="viridis", showfliers=False,
            showmeans=True); set_box_alpha(ax, 0.5)
sns.swarmplot(x="label", y="per", hue="loc",
              hue_order=hue_order, data=df.query("classification == True"), order=["pkg_bk", "pkg_dk", "pkg_tremor"],
              palette="viridis", alpha=0.5, dodge=True, size=2.5)
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.title("Region-wise performances")
# 
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "region_wise_performances_classification.pdf"))
plt.show(block=True)

plt.figure(figsize=(5, 5), dpi=300)
ax = sns.boxplot(x="label", y="per", hue="loc", hue_order=hue_order, order=["pkg_bk", "pkg_dk", "pkg_tremor"],
            data=df.query("classification == False"), palette="viridis", showfliers=False,
            showmeans=True); set_box_alpha(ax, 0.5)
sns.swarmplot(x="label", y="per", hue="loc",
              hue_order=hue_order, data=df.query("classification == False"),order=["pkg_bk", "pkg_dk", "pkg_tremor"],
              palette="viridis", alpha=0.5, dodge=True, size=2.5)
plt.ylabel("Correlation coefficient")
plt.title("Region-wise performances")
plt.gca().spines['right'].set_visible(False)
plt.gca().spines['top'].set_visible(False)
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "region_wise_performances_regression.pdf"))
plt.show(block=True)

# Surfice plot ecog
# x_ecog = list(
#         np.abs(df.query ("loc == 'MC' or loc == 'SC'").query("classification == False").query("label == 'pkg_bk'")["x"].values.round(2))
#     )
# print(x_ecog)
# [-27.51, 37.47, -24.64, 37.89, -26.23, 36.03, -21.25, 31.18, -22.99, 27.1, 25.71, -22.37, 30.78, -26.88, 26.83, 27.97, 26.04, -17.79, 26.48, 23.53, -21.97, -21.39, 35.51, 22.26, 32.63, -27.31, 37.24, 27.52, 29.99, 32.37, -26.31, 30.33, -22.09, -28.03, 28.6, -27.56, 26.31, -19.68, -26.25, -32.14, 31.19, -34.44, -32.16, 32.03, -32.5, 29.34, -30.97, -32.27, -21.77, -25.36, 29.63, -35.65, -21.69, -24.71, 27.9, -31.9, -25.05]
# y_ecog = list(
#         df.query ("loc == 'MC' or loc == 'SC'").query("classification == False").query("label == 'pkg_bk'")["y"].values.round(2)
#     )
# [-13.05, -19.26, -17.89, -13.46, -32.24, -10.62, -38.27, -33.71, -10.96, -34.21, -23.14, -20.22, -13.75, -36.42, -15.03, -11.59, -11.14, -33.15, -32.36, -20.05, -12.9, -22.88, -28.62, -35.3, -13.53, -43.17, -16.8, -15.44, -33.82, -10.77, -32.49, -17.93, -12.71, -15.52, -30.92, -21.41, -38.5, -33.17, -36.66, -33.98, -3.05, -23.04, -12.49, -12.5, -13.81, -37.95, -17.13, -13.04, -37.14, -31.78, -15.74, -20.14, -34.33, -21.2, -37.75, -39.57, -13.66]

# z_ecog = list(
#         df.query ("loc == 'MC' or loc == 'SC'").query("classification == False").query("label == 'pkg_bk'")["z"].values.round(2)
#     )
# [72.47, 66.55, 77.33, 71.01, 74.31, 65.45, 78.6, 74.8, 71.78, 72.81, 74.14, 73.75, 71.58, 76.46, 73.18, 73.29, 72.45, 78.85, 74.0, 76.69, 76.09, 75.87, 71.79, 77.2, 72.34, 72.13, 69.71, 75.28, 73.6, 73.36, 71.15, 71.87, 75.37, 76.03, 76.52, 70.71, 72.47, 76.63, 76.71, 72.05, 67.66, 71.48, 71.17, 69.78, 70.56, 74.2, 72.69, 70.18, 74.81, 74.87, 72.0, 71.05, 77.72, 74.4, 73.02, 71.12, 74.91]

# per_ecog = list(
#         np.clip(df.query ("loc == 'MC' or loc == 'SC'").query("classification == False").query("label == 'pkg_bk'")["per"].values.round(2), 0.2,0.9)
#     )
# print(per_ecog)
# [0.79, 0.65, -0.15, 0.34, 0.75, 0.55, 0.55, 0.44, 0.27, 0.25, 0.83, 0.08, 0.22, 0.84, 0.83, 0.82, 0.76, 0.81, 0.8, 0.75, 0.79, 0.61, 0.58, 0.68, 0.84, 0.65, 0.62, 0.63, 0.82, -0.54, 0.79, 0.77, 0.76, 0.74, 0.65, 0.83, 0.77, 0.78, 0.52, 0.54, -0.36, 0.84, 0.46, -0.34, 0.81, 0.48, 0.76, 0.78, 0.62, 0.76, 0.48, 0.8, 0.72, 0.73, 0.55, 0.88, 0.6]
# print([0.5 for _ in range(len(per_ecog))])
# [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

def print_surfice(x, y, z, per, per_):
    print("BEGIN")
    print(f"\tNODECREATE('',")
    print(f"{x},")
    print(f"{y},")
    print(f"{z},")
    print(f"{per},")
    print(f"{per_}")
    print("\t);")
    print("END.")

# print_surfice(x_ecog, y_ecog, z_ecog, per_ecog, [0.5 for _ in range(len(per_ecog))])

print_surfice(
     list(
        np.abs(df.query ("loc == 'MC' or loc == 'SC'").query("classification == True").query("label == 'pkg_dk'")["x"].values.round(2))
    ),
    list(
        df.query ("loc == 'MC' or loc == 'SC'").query("classification == True").query("label == 'pkg_dk'")["y"].values.round(2)
    ),
    list(
        df.query ("loc == 'MC' or loc == 'SC'").query("classification == True").query("label == 'pkg_dk'")["z"].values.round(2)
    ),
    list(
        np.clip(np.abs(df.query ("loc == 'MC' or loc == 'SC'").query("classification == True").query("label == 'pkg_dk'")["per"].values.round(2))
    , 0.5, 0.9)),
    [0.5 for _ in range(len(list(
        np.abs(df.query ("loc == 'MC' or loc == 'SC'").query("classification == True").query("label == 'pkg_dk'")["per"].values.round(2))
    )))]
)

print_surfice(
     list(
        np.abs(df.query ("loc == 'MC' or loc == 'SC'").query("classification == True").query("label == 'pkg_bk'")["x"].values.round(2))
    ),
    list(
        df.query ("loc == 'MC' or loc == 'SC'").query("classification == True").query("label == 'pkg_bk'")["y"].values.round(2)
    ),
    list(
        df.query ("loc == 'MC' or loc == 'SC'").query("classification == True").query("label == 'pkg_bk'")["z"].values.round(2)
    ),
    list(
        np.clip(np.abs(df.query ("loc == 'MC' or loc == 'SC'").query("classification == True").query("label == 'pkg_bk'")["per"].values.round(2))
    , 0.3, 0.9)),
    [0.5 for _ in range(len(list(
        np.abs(df.query ("loc == 'MC' or loc == 'SC'").query("classification == True").query("label == 'pkg_bk'")["per"].values.round(2))
    )))]
)


# print_surfice(
#      list(
#         np.abs(df.query ("loc == 'MC' or loc == 'SC'").query("classification == True").query("label == 'pkg_tremor'")["x"].values.round(2))
#     ),
#     list(
#         df.query ("loc == 'MC' or loc == 'SC'").query("classification == True").query("label == 'pkg_tremor'")["y"].values.round(2)
#     ),
#     list(
#         df.query ("loc == 'MC' or loc == 'SC'").query("classification == True").query("label == 'pkg_tremor'")["z"].values.round(2)
#     ),
#     list(
#         np.clip(np.abs(df.query ("loc == 'MC' or loc == 'SC'").query("classification == True").query("label == 'pkg_tremor'")["per"].values.round(2))
#     , 0.5, 0.9)),
#     [0.5 for _ in range(len(list(
#         np.abs(df.query ("loc == 'MC' or loc == 'SC'").query("classification == True").query("label == 'pkg_tremor'")["per"].values.round(2))
#     )))]
# )

# BEGIN
# 	NODECREATE('',
# [27.51, 37.47, 24.64, 37.89, 26.23, 36.03, 21.25, 31.18, 22.99, 27.1, 25.71, 22.37, 30.78, 26.88, 26.83, 27.97, 26.04, 17.79, 26.48, 23.53, 21.97, 21.39, 35.51, 22.26, 32.63, 27.31, 37.24, 27.52, 29.99, 32.37, 26.31, 30.33, 22.09, 28.03, 28.6, 27.56, 26.31, 19.68, 26.25, 32.14, 31.19, 34.44, 32.16, 32.03, 32.5, 29.34, 30.97, 32.27, 21.77, 25.36, 29.63, 35.65, 21.69, 24.71, 27.9, 31.9, 25.05],        [-13.05, -19.26, -17.89, -13.46, -32.24, -10.62, -38.27, -33.71, -10.96, -34.21, -23.14, -20.22, -13.75, -36.42, -15.03, -11.59, -11.14, -33.15, -32.36, -20.05, -12.9, -22.88, -28.62, -35.3, -13.53, -43.17, -16.8, -15.44, -33.82, -10.77, -32.49, -17.93, -12.71, -15.52, -30.92, -21.41, -38.5, -33.17, -36.66, -33.98, -3.05, -23.04, -12.49, -12.5, -13.81, -37.95, -17.13, -13.04, -37.14, -31.78, -15.74, -20.14, -34.33, -21.2, -37.75, -39.57, -13.66],
# 	[72.47, 66.55, 77.33, 71.01, 74.31, 65.45, 78.6, 74.8, 71.78, 72.81, 74.14, 73.75, 71.58, 76.46, 73.18, 73.29, 72.45, 78.85, 74.0, 76.69, 76.09, 75.87, 71.79, 77.2, 72.34, 72.13, 69.71, 75.28, 73.6, 73.36, 71.15, 71.87, 75.37, 76.03, 76.52, 70.71, 72.47, 76.63, 76.71, 72.05, 67.66, 71.48, 71.17, 69.78, 70.56, 74.2, 72.69, 70.18, 74.81, 74.87, 72.0, 71.05, 77.72, 74.4, 73.02, 71.12, 74.91],
# [0.79, 0.65, 0.2, 0.34, 0.75, 0.55, 0.55, 0.44, 0.27, 0.25, 0.83, 0.2, 0.22, 0.84, 0.83, 0.82, 0.76, 0.81, 0.8, 0.75, 0.79, 0.61, 0.58, 0.68, 0.84, 0.65, 0.62, 0.63, 0.82, 0.2, 0.79, 0.77, 0.76, 0.74, 0.65, 0.83, 0.77, 0.78, 0.52, 0.54, 0.2, 0.84, 0.46, 0.2, 0.81, 0.48, 0.76, 0.78, 0.62, 0.76, 0.48, 0.8, 0.72, 0.73, 0.55, 0.88, 0.6],
#         [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]);
# END.

