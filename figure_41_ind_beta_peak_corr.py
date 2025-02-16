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
df_all = pd.read_csv('/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_rmap/all_ch_renamed_no_rmap.csv')

subs = df_all["sub"].unique()

ind_peaks = {
    "rcs02l" : 20,
    "rcs02r" : 18,
    "rcs03l" : 13.5,
    "rcs05l" : 25,
    "rcs05r" : 25,
    "rcs06l" : 20,
    "rcs06r" : 18,
    "rcs07l" : 20,
    "rcs07r" : 20,
    "rcs08l" : 20,
    "rcs08r" : 20,
    "rcs09l" : 23,
    "rcs09r" : 23,
    "rcs10l" : 27,
    "rcs10r" : 30,
    "rcs11l" : 25,
    "rcs11r" : 17,
    "rcs12l" : 27,
    "rcs12r" : 20,
    "rcs14l" : 25,
    "rcs15l" : 18,
    "rcs15r" : 18,
    "rcs17l" : 25,
    "rcs17r" : 27,
    "rcs18l" : 23,
    "rcs18r" : 23,
    "rcs19l" : 22,
    "rcs19r" : 25,
    "rcs20l" : 18,
    "rcs20r" : 17,
}

ind_peaks_short = {
    "rcs02l" : 20,
    "rcs02r" : 18,
    "rcs03l" : 27,
    "rcs05l" : 27,
    "rcs05r" : 25,  # NONE
    "rcs06l" : 30,
    "rcs06r" : 18,
    "rcs07l" : 27,
    "rcs07r" : 20, # None
    "rcs08l" : 15,
    "rcs08r" : 25,
    "rcs09l" : 22,
    "rcs09r" : 22,
    "rcs10l" : 27,
    "rcs10r" : 27,
    "rcs11l" : 27, # DOUBLE PEAK
    "rcs11r" : 17,
    "rcs12l" : 27,
    "rcs12r" : 20, # NONE
    "rcs14l" : 25, 
    "rcs15l" : 18,
    "rcs15r" : 18,
    "rcs17l" : 30,
    "rcs17r" : 30,
    "rcs18l" : 23, # NONE
    "rcs18r" : 23,
    "rcs19l" : 22,
    "rcs19r" : 25,
    "rcs20l" : 18, # NONE
    "rcs20r" : 17,
}

subs_no_peak = ["rcs05r", "rcs07r", "rcs12r", "rcs18l", "rcs20l"]

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
PLOT_IND_EXAMPLE = False
if PLOT_IND_EXAMPLE:
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
    #plt.savefig(os.path.join(PATH_FIGURES, "figure_37_beta_prediction_example.pdf"))
    plt.show(block=True)



performance_data = []
PLOT_ = False
pdf_path = "beta_prediction_plots.pdf"
#with PdfPages(pdf_path) as pdf:
for sub in subs:
    #sub = "rcs08r"
    df_sub = df_all[df_all["sub"] == sub]
    
    low_beta = df_sub["ch_subcortex_1_fft_low beta_mean_mean"]
    high_beta = df_sub["ch_subcortex_1_fft_high beta_mean_mean"]
    range_5Hz = np.arange(ind_peaks_short[sub]-2.5, ind_peaks_short[sub]+2.5, 1).astype(int)
    ind_beta = df_sub[[f'ch_subcortex_1_welch_psd_{i}_mean' for i in range_5Hz]].mean(axis=1) # this is normalized
    all_beta = df_sub[[f'ch_subcortex_1_welch_psd_{i}_mean' for i in np.arange(13, 36, 1)]].mean(axis=1)
    pkg_dk = df_sub["pkg_dk"]
    pkg_bk = df_sub["pkg_bk"]
    pkg_tremor = df_sub["pkg_tremor"]
    idx_not_nan = ~np.isnan(pkg_dk) & ~np.isnan(pkg_bk) & ~np.isnan(pkg_tremor) & ~np.isnan(low_beta) & ~np.isnan(high_beta) & ~np.isnan(ind_beta) & ~np.isnan(all_beta)

    low_beta = low_beta[idx_not_nan]
    high_beta = high_beta[idx_not_nan]
    ind_beta = ind_beta[idx_not_nan]
    all_beta = all_beta[idx_not_nan]
    pkg_dk = pkg_dk[idx_not_nan]
    pkg_bk = pkg_bk[idx_not_nan]
    pkg_tremor = pkg_tremor[idx_not_nan]
    # z-score normalize each
    low_beta = stats.zscore(low_beta.values)
    high_beta = stats.zscore(high_beta.values)
    ind_beta = stats.zscore(ind_beta.values)
    all_beta = stats.zscore(all_beta.values)
    pkg_dk = stats.zscore(pkg_dk.values)
    pkg_bk = stats.zscore(pkg_bk.values)
    pkg_tremor = stats.zscore(pkg_tremor.values)

    if PLOT_:
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
        # Save the current figure to the PDF
        #pdf.savefig()
        plt.close()
    # Save performance data
    for pkg, label in zip([pkg_dk, pkg_bk, pkg_tremor], ["Dyskinesia", "Bradykinesia", "Tremor"]):
        performance_data.append({
            "subject": sub,
            "condition": label,
            "low_beta_corr": np.corrcoef(low_beta, pkg)[0, 1],
            "high_beta_corr": np.corrcoef(high_beta, pkg)[0, 1],
            "ind_beta_corr": np.corrcoef(ind_beta, pkg)[0, 1],
            "all_beta_corr": np.corrcoef(all_beta, pkg)[0, 1]
        })

# Convert performance data to DataFrame
performance_df = pd.DataFrame(performance_data)
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

res_df["model"] = "ML"
performance_df["pkg_decode_label"] = performance_df["condition"].replace({"Dyskinesia": "pkg_dk", "Bradykinesia": "pkg_bk", "Tremor": "pkg_tremor"}) 
# pivot table that low_beta_corr, high_beta_corr, ind_beta_corr are in the same column
performance_df = performance_df.melt(id_vars=["subject", "condition", "pkg_decode_label"], value_vars=["low_beta_corr", "high_beta_corr", "ind_beta_corr", "all_beta_corr"], var_name="beta", value_name="corr")
performance_df = performance_df.rename(columns={"subject": "sub"})
performance_df = performance_df.drop("condition", axis=1)
# rename model to beta in res_df
res_df = res_df.rename(columns={"model": "beta"})
# remove CLASSIFICATION column
res_df = res_df.drop("CLASSIFICATION", axis=1)
# rename per to to corr
res_df = res_df.rename(columns={"per": "corr"})
# drop loc
res_df = res_df.drop("loc", axis=1)

# concatenate the two dataframes
df_concat = pd.concat([res_df, performance_df], axis=0)
df_concat["corr"] = df_concat["corr"].abs()
df_concat["has_peak"] = df_concat["sub"].apply(lambda x: x not in subs_no_peak)

plt.figure(figsize=(4, 7))
df_plt = df_concat.query("beta != 'ML'")
#for idx, pkg_decode_label in enumerate(["pkg_bk", "pkg_dk", "pkg_tremor"]):
pkg_decode_label = "pkg_bk"
#plt.subplot(1, 3, idx+1)
sns.boxplot(data=df_plt.query(f"pkg_decode_label == '{pkg_decode_label}'"), x="beta", y="corr", hue="has_peak", palette="viridis", showmeans=True, showfliers=False, boxprops=dict(alpha=0.5))
sns.swarmplot(data=df_plt.query(f"pkg_decode_label == '{pkg_decode_label}'"), x="beta", y="corr", hue="has_peak", dodge=True, palette="viridis", alpha=0.9, s=5)
plt.ylabel("Correlation coefficient")
plt.title(pkg_decode_label)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(PATH_FIGURES, "figure_41_beta_comparison_hasPeaks.pdf"))

from py_neuromodulation import nm_stats
df_test = df_concat.query("pkg_decode_label == 'pkg_bk'")
for beta_val in ["low_beta_corr", "high_beta_corr", "ind_beta_corr", "all_beta_corr"]:
    _, p = nm_stats.permutationTest(df_test.query(f"beta == '{beta_val}' and has_peak == True")["corr"].values,
                                        df_test.query(f"beta == '{beta_val}' and has_peak == False")["corr"].values,
                                        False, None, p=5000)
    print(f"{beta_val} p-value: {p}")


plt.figure(figsize=(4.5, 7))
df_plt = df_concat.query("beta == 'all_beta_corr' or beta == 'ind_beta_corr' or beta == 'ML'")
sns.boxplot(data=df_plt, x="pkg_decode_label", y="corr", hue="beta", palette="viridis", showmeans=True, showfliers=False,
            boxprops=dict(alpha=0.5), order=["pkg_bk", "pkg_dk", "pkg_tremor"], hue_order=["ML", "all_beta_corr", "ind_beta_corr"])
sns.swarmplot(data=df_plt, x="pkg_decode_label", y="corr", hue="beta", dodge=True, palette="viridis", alpha=0.9, s=5,
                order=["pkg_bk", "pkg_dk", "pkg_tremor"], hue_order=["ML", "all_beta_corr", "ind_beta_corr"])
plt.ylabel("Correlation coefficient")
plt.title("Beta prediction")
plt.savefig(os.path.join(PATH_FIGURES, "figure_41_beta_comparison_to_ML.pdf"))
#plt.show(block=True)

from py_neuromodulation import nm_stats

for pkg_decode_label in ["pkg_bk", "pkg_dk", "pkg_tremor"]:
    for beta_val in ["ML", "all_beta_corr", "ind_beta_corr"]:
        for beta_val2 in ["ML", "all_beta_corr", "ind_beta_corr"]:
            if beta_val == beta_val2:
                continue
            _, p = nm_stats.permutationTest_relative(df_concat.query(f"beta == '{beta_val}' and pkg_decode_label == '{pkg_decode_label}'")["corr"].values,
                                            df_concat.query(f"beta == '{beta_val2}' and pkg_decode_label == '{pkg_decode_label}'")["corr"].values,
                                            False, None, p=5000)
            print(f"{pkg_decode_label} {beta_val} vs {beta_val2} p-value: {p}")
            print(f"mean: {df_concat.query(f'beta == "{beta_val}" and pkg_decode_label == "{pkg_decode_label}"')["corr"].mean()} ± {df_concat.query(f'beta == "{beta_val}" and pkg_decode_label == "{pkg_decode_label}"')["corr"].std()}")
            print(f"mean: {df_concat.query(f'beta == "{beta_val2}" and pkg_decode_label == "{pkg_decode_label}"')["corr"].mean()} ± {df_concat.query(f'beta == "{beta_val2}" and pkg_decode_label == "{pkg_decode_label}"')["corr"].std()}")
            print("")

plt.figure(figsize=(10, 7))
sns.boxplot(data=df_concat, x="pkg_decode_label", y="corr", hue="beta", palette="viridis", showmeans=True, showfliers=False, boxprops=dict(alpha=0.5), order=["pkg_bk", "pkg_dk", "pkg_tremor"])
sns.swarmplot(data=df_concat, x="pkg_decode_label", y="corr", hue="beta", dodge=True, palette="viridis", alpha=0.9, s=5,
              order=["pkg_bk", "pkg_dk", "pkg_tremor"])
plt.ylabel("Correlation coefficient")
plt.title("Beta prediction")
# print mean and std of corr above each box
mean_corr = df_concat.groupby(["pkg_decode_label", "beta"])["corr"].mean()
std_corr = df_concat.groupby(["pkg_decode_label", "beta"])["corr"].std()
plt.show(block=True)

plt.figure(figsize=(10, 7))
plt.subplot(121)
sns.boxplot(data=df_concat.query("beta == 'high_beta_corr'"), x="pkg_decode_label", y="corr", hue="has_peak", palette="viridis", showmeans=True, showfliers=False, boxprops=dict(alpha=0.5), order=["pkg_bk", "pkg_dk", "pkg_tremor"])
sns.swarmplot(data=df_concat.query("beta == 'high_beta_corr'"), x="pkg_decode_label", y="corr", hue="has_peak", dodge=True, palette="viridis", alpha=0.9, s=5, order=["pkg_bk", "pkg_dk", "pkg_tremor"])
plt.ylabel("Correlation coefficient")
plt.title("High beta")
plt.subplot(122)
sns.boxplot(data=df_concat.query("beta == 'ML'"), x="pkg_decode_label", y="corr", hue="has_peak", palette="viridis", showmeans=True, showfliers=False, boxprops=dict(alpha=0.5), order=["pkg_bk", "pkg_dk", "pkg_tremor"])
sns.swarmplot(data=df_concat.query("beta == 'ML'"), x="pkg_decode_label", y="corr", hue="has_peak", dodge=True, palette="viridis", alpha=0.9, s=5, order=["pkg_bk", "pkg_dk", "pkg_tremor"])
plt.ylabel("Correlation coefficient")
plt.title("ML")
plt.tight_layout()
plt.show(block=True)


# rename low_beta_corr to per
performance_df = performance_df.rename(columns={"low_beta_corr": "per"})
performance_df["model"] = "Low beta prediction"
performance_df["loc"] = "stn"

# concatenate the two dataframes
df_concat = pd.concat([res_df, performance_df], axis=0)
# take abs value of per
df_concat["per"] = df_concat["per"].abs()

performance_df.to_csv("beta_prediction_performance.csv", index=False)



plt.figure(figsize=(10, 7))
sns.boxplot(data=performance_df, x="condition", y="corr", hue="beta", palette="viridis", showmeans=True, showfliers=False)
sns.swarmplot(data=performance_df, x="condition", y="corr", hue="beta", dodge=True, palette="viridis", alpha=0.9, s=2)
plt.ylabel("Correlation coefficient")
plt.title("Beta prediction")
plt.show(block=True)


plt.figure(figsize=(10, 7))
plt.subplot(131)
sns.boxplot(data=performance_df, x="condition", y="low_beta_corr", palette="viridis", showmeans=True, showfliers=False)
sns.swarmplot(data=performance_df, x="condition", y="low_beta_corr", dodge=False, palette="viridis", alpha=0.9, s=2)
plt.ylabel("Correlation coefficient")
plt.title("Low beta")
plt.subplot(132)
sns.boxplot(data=performance_df, x="condition", y="high_beta_corr", palette="viridis", showmeans=True, showfliers=False)
sns.swarmplot(data=performance_df, x="condition", y="high_beta_corr", dodge=False, palette="viridis", alpha=0.9, s=2)
plt.ylabel("Correlation coefficient")
plt.title("High beta")
plt.subplot(133)
sns.boxplot(data=performance_df, x="condition", y="ind_beta_corr", palette="viridis", showmeans=True, showfliers=False)
sns.swarmplot(data=performance_df, x="condition", y="ind_beta_corr", dodge=False, palette="viridis", alpha=0.9, s=2)
plt.ylabel("Correlation coefficient")
plt.title("Individual beta")
plt.tight_layout()
plt.show(block=True)



plt.figure(figsize=(10, 7))
sns.boxplot(data=res_df, x="pkg_decode_label", y="per", palette="viridis", showmeans=True, showfliers=False)
sns.swarmplot(data=res_df, x="pkg_decode_label", y="per", dodge=False, palette="viridis", alpha=0.9, s=2)
plt.ylabel("Correlation coefficient")
plt.title("STN")
plt.show(block=True)

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

