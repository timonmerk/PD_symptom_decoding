import pandas as pd
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import os

ESTIMASTE_QUASI_PROBA = True

# df_out_predictions_true_class_zs

if ESTIMASTE_QUASI_PROBA:

    df = pd.read_csv("publication_figures/df_out_predictions_true.csv")

    df_ = []
    for sub in df["sub"].unique():
        df_sub = df[df["sub"] == sub]
        for label in df_sub["label"].unique():
            df_sub_label = df_sub[df_sub["label"] == label]
            if label == "pkg_dk":
                label_pr_normed = (df_sub_label["pr"].copy() / df_sub_label["pr"].max())
                pr_class = label_pr_normed > 0.05
                pr_quasi_proba = label_pr_normed / 0.05
                
                true_normed = df_sub_label["true"] / df_sub_label["true"].max()
                true_class = true_normed > 0.05
                true_quasi_proba = true_normed / 0.05

            elif label == "pkg_tremor":
                label_thr = 5
                pr_class = df_sub_label["pr"] > label_thr
                pr_quasi_proba = df_sub_label["pr"] / label_thr
                true_class = df_sub_label["true"] > label_thr
                true_quasi_proba = df_sub_label["true"] / label_thr
            elif label == "pkg_bk":
                label_thr = 20
                pr_class = df_sub_label["pr"] > label_thr
                pr_quasi_proba = df_sub_label["pr"] / label_thr
                true_class = df_sub_label["true"] > label_thr
                true_quasi_proba = df_sub_label["true"] / label_thr
            
            pr_quasi_proba = np.clip(pr_quasi_proba, 0, 1)
            true_quasi_proba = np.clip(true_quasi_proba, 0, 1)

            quantized_pr = pd.qcut(df_sub_label["pr"], 10, labels=False, duplicates='drop')
            quantized_true = pd.qcut(df_sub_label["true"], 10, labels=False, duplicates='drop')

            df_cp = df_sub_label[["pkg_dt", "sub", "label"]].copy()
            df_cp["pr"] = df_sub_label["pr"]
            df_cp["true"] = df_sub_label["true"]
            # df_cp["pr_class"] = pr_class
            # df_cp["pr_quasi_proba"] = pr_quasi_proba
            # df_cp["true_quasi_proba"] = true_quasi_proba
            # df_cp["true_class"] = true_class
            # df_cp["quantized_pr"] = quantized_pr
            # df_cp["quantized_true"] = quantized_true
            df_cp["pr_zs"] = stats.zscore(df_sub_label["pr"])
            df_.append(
                df_cp
            ) 
    df_all = pd.concat(df_, ignore_index=True)
    df_all.to_csv("publication_figures/df_out_predictions_true_class_quasi_proba.csv", index=False)
else:
    df_all = pd.read_csv("publication_figures/df_out_predictions_true_class_quasi_proba.csv")

print(df_all)
df_pivot = df_all.pivot_table(index=["sub", "pkg_dt"], columns="label", values=["pr", "true", "pr_zs"], aggfunc='first')
#df_pivot = df_all.pivot_table(index=["sub", "pkg_dt"], columns="label", values=["pr", "true", "pr_class", "true_class", "quantized_pr", "quantized_true", "pr_quasi_proba", "true_quasi_proba"], aggfunc='first')
# reset index to have a flat dataframe
df_pivot = df_pivot.reset_index()
# reset dual column names
df_pivot.columns = ['_'.join(col).strip() if col[1] else col[0] for col in df_pivot.columns.values]
# save the pivoted dataframe
df_pivot["ratio_pr_bk_tremor"] = df_pivot["pr_pkg_bk"] / df_pivot["pr_pkg_tremor"]
df_pivot["ratio_proba_pr_bk_tremor"] = df_pivot["pr_quasi_proba_pkg_bk"] / df_pivot["pr_quasi_proba_pkg_tremor"]
df_pivot["ratio_quantized_pr_bk_tremor"] = df_pivot["quantized_pr_pkg_bk"] / df_pivot["quantized_pr_pkg_tremor"]

df_pivot.to_csv("publication_figures/df_out_predictions_true_class_zs.csv")

plt.figure(figsize=(10, 3))
sub = "rcs02l"
df_sub = df_all[df_all["sub"] == sub]
for label in df_sub["label"].unique():
    df_sub_label = df_sub[df_sub["label"] == label]
    plt.plot(df_sub_label["pkg_dt"], df_sub_label["pr_zs"], label=f"pr_{label}")
plt.legend()
plt.title(f"Sub: {sub}")
plt.xlabel("Time")
plt.ylabel("Quantized Prediction")
# remove x labels
plt.xticks([])

# note, they are not exclusive
# get the percentage of classes being 1 for each label and each sub
df_perc = df_all.groupby(["sub", "label"])["true_class"].mean().reset_index()
plt.figure(figsize=(10, 5))
sns.barplot(data=df_perc, x="sub", y="true_class", hue="label")
plt.title("Percentage of classes being 1 for each label and each sub")
plt.xlabel("Label")
plt.xticks(rotation=90)
plt.tight_layout()

df_perc = df_all.groupby(["sub", "label"])["pr_class"].mean().reset_index()
plt.figure(figsize=(10, 5))
sns.barplot(data=df_perc, x="sub", y="pr_class", hue="label")
plt.title("Percentage of classes being 1 for each label and each sub")
plt.xlabel("Label")
plt.xticks(rotation=90)
plt.tight_layout()