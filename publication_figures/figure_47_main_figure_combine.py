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

def read_file(PATH_READ, CLASSIFICATION, label_name, loc_):
    with open(PATH_READ, "rb") as f:
        d_out = pickle.load(f)
    data = []
    #for CLASSIFICATION in d_out.keys():
    if CLASSIFICATION:
        per_ = "ba"
    else:
        per_ = "corr_coeff"
    #    for pkg_decode_label in d_out[CLASSIFICATION].keys():
    #        for loc in d_out[CLASSIFICATION][pkg_decode_label].keys():
    for sub in d_out.keys():
        data.append({
            "CLASSIFICATION": CLASSIFICATION,
            "per": d_out[sub][per_],
            "sub": sub,
            "pkg_decode_label": label_name,
            "loc": loc_
        })

    df = pd.DataFrame(data)
    return df

PATH_PER = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/paper_per/without_night'

l_all = []
for label_name in ["pkg_dk", "pkg_tremor", "pkg_bk"]:
    for CLASSIFICATION in [False, True]:
        for loc_ in ["ecog", "stn", "ecog_stn"]:   # f"LOHO_main_{label_name}_CLASS_{CLASSIFICATION}_loc_{loc_}_withpsd.pk
            PATH_NAME = f"LOHO_main_{label_name}_CLASS_{CLASSIFICATION}_loc_{loc_}_withpsd.pkl"
            PATH_READ = os.path.join(PATH_PER, PATH_NAME)
            df = read_file(PATH_READ, CLASSIFICATION, label_name, loc_)
            l_all.append(df)
df = pd.concat(l_all)

df.groupby(["CLASSIFICATION", "pkg_decode_label", "loc"])["per"].agg([np.mean, np.std]).round(2)

df.to_csv(os.path.join(PATH_PER, "df_main.csv"))