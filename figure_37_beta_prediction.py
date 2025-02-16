import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

PATH_FEATURES = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_normalized_10s_window_length/480"
PATH_FIGURES = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf'
df_all = pd.read_csv(os.path.join(PATH_FEATURES, "all_merged_normed.csv"), index_col=0)

[c for c in df_all.columns if "subcortex" in c and "fft" in c and "beta" in c]

subs = df_all["sub"].unique()

import pickle
PATH_PER = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per'
OUT_FILE = "LOHO_ALL_LABELS_ALL_GROUPS_normed_480.pkl"
PATH_FIGURES = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf'
#OUT_FILE = "LOHO_ALL_LABELS_ALL_GROUPS_nonorm.pkl"
PATH_READ = os.path.join(PATH_PER, OUT_FILE)

with open(PATH_READ, "rb") as f:
    d_out = pickle.load(f)

# 'ch_subcortex_fft_low beta_mean_mean'
# 'ch_subcortex_fft_high beta_mean_mean'

# Plot example prediction beta vs decoding
sub = "rcs02r"
df_sub = df_all[df_all["sub"] == sub]
low_beta = df_sub["ch_subcortex_fft_low beta_mean_mean"].values
low_beta = stats.zscore(low_beta).clip(-2,2.5)
pkg_bk = df_sub["pkg_bk"].values
pkg_bk = stats.zscore(pkg_bk).clip(-2, 2.5)

window_size = 5
weights = np.ones(window_size) / window_size  # Equal weights

corr_ = np.corrcoef(low_beta, pkg_bk)[0, 1]
ml_pred = d_out[False]["pkg_bk"]["ecog_stn"][sub]["pr"]
ml_pred = stats.zscore(ml_pred).clip(-2, 2.5)

plt.figure(figsize=(10, 4))
plt.subplot(211)
corr_ml = np.corrcoef(ml_pred, pkg_bk)[0, 1]
plt.plot(pkg_bk, label="Bradykinesia", color="black",)
plt.plot(low_beta*-1 , label="Low beta", color="red", alpha=0.6)
plt.legend()

plt.subplot(212)
plt.plot(pkg_bk, label="Bradykinesia", color="black",)
plt.plot(ml_pred, label="ML prediction", color="blue", alpha=0.6)
plt.legend()
plt.ylim(-2.7, 2.7)
plt.savefig(os.path.join(PATH_FIGURES, "figure_37_beta_prediction_example.pdf"))
plt.show(block=True)



performance_data = []

pdf_path = "beta_prediction_plots.pdf"
with PdfPages(pdf_path) as pdf:
    for sub in subs:
        df_sub = df_all[df_all["sub"] == sub]
        
        low_beta = df_sub["ch_subcortex_fft_low beta_mean_mean"]
        high_beta = df_sub["ch_subcortex_fft_high beta_mean_mean"]
        pkg_dk = df_sub["pkg_dk"]
        pkg_bk = df_sub["pkg_bk"]
        pkg_tremor = df_sub["pkg_tremor"]

        # z-score normalize each
        low_beta = stats.zscore(low_beta.values)
        high_beta = stats.zscore(high_beta.values)
        pkg_dk = stats.zscore(pkg_dk.values)
        pkg_bk = stats.zscore(pkg_bk.values)
        pkg_tremor = stats.zscore(pkg_tremor.values)

        plt_cnt = 1
        plt.figure(figsize=(10, 10))
        for pkg, label in zip([pkg_dk, pkg_bk, pkg_tremor], ["Dyskinesia", "Bradykinesia", "Tremor"]):
            plt.subplot(3, 2, plt_cnt)
            plt.plot(low_beta, label="Low beta")
            plt.plot(pkg, label=label)
            corr_ = np.corrcoef(low_beta, pkg)[0, 1]
            plt.title(f"Low beta vs {label} - corr: {corr_:.2f}")
            plt.legend()
            plt_cnt += 1
            plt.subplot(3, 2, plt_cnt)
            plt.plot(high_beta, label="High beta")
            plt.plot(pkg, label=label)
            corr_ = np.corrcoef(high_beta, pkg)[0, 1]
            plt.title(f"High beta vs {label} - corr: {corr_:.2f}")
            plt_cnt += 1
            plt.legend()
            plt.tight_layout()
            plt.suptitle(f"Subject: {sub}")

            # Save performance data
            performance_data.append({
                "subject": sub,
                "condition": label,
                "low_beta_corr": np.corrcoef(low_beta, pkg)[0, 1],
                "high_beta_corr": np.corrcoef(high_beta, pkg)[0, 1]
            })

        # Save the current figure to the PDF
        pdf.savefig()
        plt.close()

# Convert performance data to DataFrame
performance_df = pd.DataFrame(performance_data)
performance_df.to_csv("beta_prediction_performance.csv", index=False)

# make a scatter plot of the performance data
plt.figure(figsize=(10, 7))
plt.subplot(121)
sns.boxplot(data=performance_df, x="condition", y="low_beta_corr", palette="viridis", showmeans=True, showfliers=False)
sns.swarmplot(data=performance_df, x="condition", y="low_beta_corr", dodge=False, palette="viridis", alpha=0.9, s=2)
plt.ylabel("Correlation coefficient")
plt.title("Low beta")
plt.subplot(122)
sns.boxplot(data=performance_df, x="condition", y="high_beta_corr", palette="viridis", showmeans=True, showfliers=False)
sns.swarmplot(data=performance_df, x="condition", y="high_beta_corr", dodge=False, palette="viridis", alpha=0.9, s=2)
plt.ylabel("Correlation coefficient")
plt.title("High beta")
plt.tight_layout()
plt.show(block=True)


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
res_df = df[df["CLASSIFICATION"] == False].query("loc == 'ecog_stn'")

plt.figure(figsize=(10, 7))
sns.boxplot(data=res_df, x="pkg_decode_label", y="per", palette="viridis", showmeans=True, showfliers=False)
sns.swarmplot(data=res_df, x="pkg_decode_label", y="per", dodge=False, palette="viridis", alpha=0.9, s=2)
plt.ylabel("Correlation coefficient")
plt.title("STN")
plt.show(block=True)

res_df["model"] = "ML"
# rename condition to pkg_decode_label
performance_df["pkg_decode_label"] = performance_df["condition"].replace({"Dyskinesia": "pkg_dk", "Bradykinesia": "pkg_bk", "Tremor": "pkg_tremor"}) 
# remove condition column
performance_df = performance_df.drop("condition", axis=1)
# remame subject to sub
performance_df = performance_df.rename(columns={"subject": "sub"})
# rename low_beta_corr to per
performance_df = performance_df.rename(columns={"low_beta_corr": "per"})
performance_df["model"] = "Low beta prediction"
performance_df["loc"] = "stn"

# concatenate the two dataframes
df_concat = pd.concat([res_df, performance_df], axis=0)
# take abs value of per
df_concat["per"] = df_concat["per"].abs()

plt.figure(figsize=(10, 7), dpi=300)
sns.boxplot(data=df_concat, x="pkg_decode_label", y="per", hue="model", palette="viridis", showmeans=True, showfliers=False, boxprops=dict(alpha=0.5), order=["pkg_bk", "pkg_dk", "pkg_tremor"])
sns.swarmplot(data=df_concat, x="pkg_decode_label", y="per", hue="model", dodge=True, palette="viridis", alpha=0.9, s=5, order=["pkg_bk", "pkg_dk", "pkg_tremor"])
plt.ylabel("Correlation coefficient")
plt.title("STN")
plt.savefig(os.path.join(PATH_FIGURES, "figure_37_beta_comparison_to_ML.pdf"))
plt.show(block=True)

# take the difference between the two models for each subject and condition
df_diff = df_concat.pivot_table(index=["sub", "pkg_decode_label"], columns="model", values="per").reset_index()
df_diff["diff"] = df_diff["Low beta prediction"] - df_diff["ML"]
df_diff = df_diff.dropna()


plt.figure(figsize=(5, 7))
sns.boxplot(data=df_diff, x="pkg_decode_label", y="diff", palette="viridis", showmeans=True, showfliers=False, boxprops=dict(alpha=0.5))
sns.swarmplot(data=df_diff, x="pkg_decode_label", y="diff", dodge=False, palette="viridis", alpha=0.9, s=5)
plt.ylabel("Difference in correlation coefficient")
mean_diff = df_diff.groupby(["pkg_decode_label"])["diff"].mean()
std_diff = df_diff.groupby(["pkg_decode_label"])["diff"].std()
str_title = f"Performance differences\nBK: {mean_diff['pkg_bk'].round(2)} ± {std_diff['pkg_bk'].round(2)}\nDK: {mean_diff['pkg_dk'].round(2)} ± {std_diff['pkg_dk'].round(2)}\nTremor: {mean_diff['pkg_tremor'].round(2)} ± {std_diff['pkg_tremor'].round(2)}"

perc_bk = np.sum(df_diff.query("pkg_decode_label == 'pkg_bk'")["diff"] < 0) / len(df_diff.query("pkg_decode_label == 'pkg_bk'"))
perc_dk = np.sum(df_diff.query("pkg_decode_label == 'pkg_dk'")["diff"] < 0) / len(df_diff.query("pkg_decode_label == 'pkg_dk'"))
perc_tremor = np.sum(df_diff.query("pkg_decode_label == 'pkg_tremor'")["diff"] < 0) / len(df_diff.query("pkg_decode_label == 'pkg_tremor'"))
str_title += f"\nPercentage ML better BK: {perc_bk:.2f} DK: {perc_dk:.2f} Tremor: {perc_tremor:.2f}"

plt.title(str_title)
plt.show(block=True)

