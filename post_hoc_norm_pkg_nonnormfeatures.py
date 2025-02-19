import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np

df_features = pd.read_csv('/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_std_10s_window_length/all_merged_preprocessed_with_condition.csv')
df_features["pkg_dt"] = pd.to_datetime(df_features["pkg_dt"], utc=True).dt.tz_convert("US/Pacific")

# compute a running average of the pkg scores with a time window of plus minus 5 minutes
df_features["pkg_dk_5min"] = None
df_features["pkg_bk_5min"] = None
df_features["pkg_tremor_5min"] = None

def norm_subject(sub):
    df_sub = df_features[df_features["sub"] == sub]
    df_sub = df_sub.sort_values(by="pkg_dt")

    for idx, row in df_sub.iterrows():
        if idx % 100 == 0:
            print(idx)
        time_before = df_sub.loc[idx, "pkg_dt"] - pd.Timedelta(minutes=5)
        time_after = df_sub.loc[idx, "pkg_dt"] + pd.Timedelta(minutes=5)

        df_range = df_sub.query("pkg_dt >= @time_before and pkg_dt <@time_after")
        if df_range.shape[0] < 2:
            continue

        row_add = df_range[["pkg_dk", "pkg_bk", "pkg_tremor"]].mean()

        df_sub.loc[idx, "pkg_dk_5min"] = row_add["pkg_dk"]
        df_sub.loc[idx, "pkg_bk_5min"] = row_add["pkg_bk"]
        df_sub.loc[idx, "pkg_tremor_5min"] = row_add["pkg_tremor"]
    return df_sub
l_ = []
from joblib import Parallel, delayed

l_subs = Parallel(n_jobs=-1)(delayed(norm_subject)(sub) for sub in df_features["sub"].unique())
df_features_ = pd.concat(l_subs).reset_index(drop=True)

plt.figure()
plt.plot(df_features.query('sub=="rcs02r"')["pkg_dk"].values, label="check")
plt.plot(df_features_.query('sub=="rcs02r"')["pkg_dk_5min"].values, label="pkg_dk_5min")
plt.legend()
plt.show()

df_features_["pkg_dk"] = df_features_["pkg_dk_5min"]
df_features_["pkg_bk"] = df_features_["pkg_bk_5min"]
df_features_["pkg_tremor"] = df_features_["pkg_tremor_5min"]
# drop the columns
df_features_ = df_features_.drop(columns=["pkg_dk_5min", "pkg_bk_5min", "pkg_tremor_5min"])
# drop Unnamed: 0 column
df_features_ = df_features_.drop(columns=["Unnamed: 0"])
df_features_.to_csv('/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_std_10s_window_length/all_merged_preprocessed_with_condition_pkgnormed.csv')