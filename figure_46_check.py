PATH_PER = r"/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/paper_per"
import pickle
import pandas as pd
import os
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def plot_boxplot(df, x_label, y_label="Balanced accuracy", order_ = None, plt_txt = False, hide_ylabel=False):
    sns.boxplot(x=x_label, y="per", data=df, showmeans=True, showfliers=False, palette="viridis", order=order_)
    #sns.swarmplot(x="norm_window", y="ba", data=df_all, color="black", alpha=0.5, palette="viridis")
    # put the mean values as text on top of the boxplot
    means = df.groupby(x_label)["per"].mean()
    if plt_txt:
        if order_ is not None:
            for i, x_label_ in enumerate(df.groupby(x_label)["per"].mean().sort_values(ascending=True).index):
                mean = df[df[x_label] == x_label_]["per"].mean()
                plt.text(i, mean, f"{np.round(mean, 2)}", ha="center", va="bottom")
        else:
            for i, mean in enumerate(means):
                plt.text(i, mean, f"{mean:.2f}", ha="center", va="bottom")

    plt.xlabel("")
    plt.ylabel(y_label)
    plt.xticks(rotation=90)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['top'].set_visible(False)
    if hide_ylabel:
        plt.ylabel("")
    #plt.tight_layout()
    #plt.show(block=True)

def read_per_out(PATH_):
    with open(PATH_, "rb") as f:
        d_out = pickle.load(f)

    data = []

    if list(d_out.keys())[0].startswith("rcs"):
        if "pkg_bk" in PATH_:
            key_per = "corr_coeff"
            pkg_decode_label = "pkg_bk"
        elif "pkg_dk" in PATH_:
            pkg_decode_label = "pkg_dk"
            key_per = "ba"
        elif "pkg_tremor" in PATH_:
            pkg_decode_label = "pkg_tremor"
            key_per = "ba"
        for sub in d_out.keys():
            data.append({
                "sub": sub,
                "pkg_decode_label": pkg_decode_label,
                "per": d_out[sub][key_per],
            })
        df = pd.DataFrame(data)
        return df
    for pkg_decode_label in d_out.keys():
        for loc in d_out[pkg_decode_label].keys():
            for sub in d_out[pkg_decode_label][loc].keys():
                if pkg_decode_label == "pkg_bk":
                    data.append({
                        "sub" : sub,
                        "pkg_decode_label": pkg_decode_label,
                        "per": d_out[pkg_decode_label][loc][sub]["corr_coeff"],
                        #"r2" : d_out[pkg_decode_label][loc][sub]["r2"],
                        #"mae" : d_out[pkg_decode_label][loc][sub]["mae"],
                        #"mse" : d_out[pkg_decode_label][loc][sub]["mse"],
                    })
                else:
                    data.append({
                        "sub": sub,
                        "pkg_decode_label": pkg_decode_label,
                        #"f1": d_out[pkg_decode_label][loc][sub]["f1"],
                        "per": d_out[pkg_decode_label][loc][sub]["ba"],
                        
                    })

    df = pd.DataFrame(data)
    return df    

label_name = "pkg_bk"

for idx_, label_name in enumerate(["pkg_bk", "pkg_dk", "pkg_tremor"]):

    l_models = []
    for ML_ in ["LM", "XGB", "PCA_LM", "CEBRA"]:  #"CB", 
        PATH_READ = os.path.join(PATH_PER, f"d_out_ML_across_patients_{label_name}_nonorm_all_{ML_}_withpsd.pkl")
        df = read_per_out(PATH_READ)
        df["model"] = ML_
        l_models.append(df)
    df_models = pd.concat(l_models)

    mod_files = [f for f in os.listdir(PATH_PER) if f"d_out_patient_across_{label_name}_feature_mod" in f]
    mods = [f[f.find("feature_mod_")+len("feature_mod_"):f.find(".pkl")] for f in mod_files]

    l_features = []
    for mod_idx, mod in enumerate(mod_files):
        PATH_READ = os.path.join(PATH_PER, mod)
        df = read_per_out(PATH_READ)
        df["feature_mod"] = mods[mod_idx]
        l_features.append(df)
    
    df_features = pd.concat(l_features, axis=0)

plt.figure()
plt.subplot(1, 1, 1)
plot_boxplot(df_models, "model", "BA",
            order_=df_models.groupby("model")["per"].mean().sort_values(ascending=True).index,
            hide_ylabel=True)

plot_boxplot(df_features, "feature_mod", "BA",
            order_=df_features.groupby("feature_mod")["per"].mean().sort_values(ascending=True).index,
            hide_ylabel=False)