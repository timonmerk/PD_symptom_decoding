import pandas as pd
import pickle 
import os
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import ast
import re



# read the X columns from pickle
with open("publication_figures/cols_ind_chs.pkl", "rb") as f:
    cols = list(pd.read_pickle(f))


cols = [col[len("ch_cortex_1_"):] if col != "h" else "h" for col in cols]

PATH_IN = "/Users/Timon/Downloads/with_importances"

#'/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/ch_used_per_sub.csv'

subs_GP = ["rcs09l", "rcs09r", "rcs10l", "rcs10r", "rcs14r", "rcs19l", "rcs19r"]

l_ = []
for f in os.listdir(PATH_IN):
    if f.endswith(".csv"):
        df = pd.read_csv(os.path.join(PATH_IN, f))
        if df.empty:
            continue
        sub = f.split("_")[4]
        ch = f.split("_")[6]
        if ch == "cortex":
            loc = "cortex"
        elif sub in subs_GP:
            loc = "GP"
        else:
            loc = "STN"
        
        df_add = df.query("classification == 0")
        df_add["loc"] = loc
        l_.append(df_add)
df = pd.concat(l_)

out_df = []
for region in ["cortex", "GP", "STN"]:
    for label in ["pkg_bk", "pkg_dk", "pkg_tremor"]:
        df_region = df.query(f"loc == '{region}' and label == '{label}'")
        importances = []
        subs = []
        for i in range(len(df_region)):
            s = df_region["feature_importances"].iloc[i]
            s_clean = re.sub(r'[\[\]\n]', ' ', s)
            vec = np.fromstring(s_clean, sep=' ')
            importances.append(vec)
            subs.append(df_region["sub"].iloc[i])
        df_importances = pd.DataFrame(importances, columns=cols)
        # remove zero columns
        #df_importances = df_importances.loc[:, (df_importances != 0).any(axis=0)]
        df_importances["loc"] = region
        df_importances["label"] = label
        df_importances["sub"] = subs
        out_df.append(df_importances)
df_ = pd.concat(out_df)

fft_mean_cols = [col for col in df_.columns if ("fft" in col and "mean_mean" in col) or ("loc" == col) or ("label" == col) or ("sub" == col)]
df_importances_fft_mean = df_[fft_mean_cols].copy()
# pivot s.t. every column excep label and loc is in a value and label column is the fft col name
df_importances_fft_mean = df_importances_fft_mean.melt(id_vars=["loc", "label", "sub"], var_name="fft_feature", value_name="importance")

plt.figure(figsize=(10, 6))
for r_i, region in enumerate(["cortex", "STN", "GP"]):
    for l_i, label in enumerate(["pkg_bk", "pkg_tremor", "pkg_dk"]):
        plt.subplot(3, 3, r_i * 3 + l_i + 1)
        df_region_label = df_importances_fft_mean.query(f"loc == '{region}' and label == '{label}'")
        mean_per = df_region_label.groupby("fft_feature")["importance"].mean()
        # min max norm
        order_ = ["fft_theta_mean_mean", "fft_alpha_mean_mean", "fft_low beta_mean_mean", "fft_high beta_mean_mean", "fft_low gamma_mean_mean", "fft_high gamma_mean_mean"]
        order_names = ["Theta", "Alpha", "Low Beta", "High Beta", "Low Gamma", "High Gamma"]
        mean_per = mean_per.reindex(order_)
        mean_per.index = order_names
        mean_per = (mean_per - mean_per.min()) / (mean_per.max() - mean_per.min())
        sns.barplot(x=mean_per.values, y=mean_per.index, palette="viridis")
        plt.title(f"{region} \n {label}")
        plt.ylabel("")
        if r_i == 2:
            plt.xlabel("Mean Feature Importance")
        else:
            plt.xlabel("")
        # turn off upper and right spines
        sns.despine(left=False, bottom=False, right=True, top=True)
plt.tight_layout()
        


# plot as the top 10 features
top_features = df_importances.mean().sort_values(ascending=False).head(10)
plt.figure(figsize=(10, 6))
sns.barplot(x=top_features.values, y=top_features.index, palette="viridis")
plt.title(f"Top 10 Feature Importances for {region}")
plt.xlabel("Mean Feature Importance")
plt.ylabel("Features")
plt.tight_layout()

# select on columns that have fft and mean in name


plt.figure(figsize=(10, 6))
sns.barplot(x=df_importances_fft_mean.mean().values, y=df_importances_fft_mean.columns, palette="viridis")
plt.title(f"FFT Mean Feature Importances for {region}")
plt.xlabel("Mean Feature Importance")
plt.ylabel("FFT Mean Features")
plt.tight_layout()

