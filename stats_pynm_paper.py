from py_neuromodulation import nm_stats
import pandas as pd

PATH_ = "/Users/Timon/Downloads/performances_cross_val_all_combined_with_CEBRA_Offset_10.csv"

df = pd.read_csv(PATH_)

for cv_type in df["Cross Validation Type"].unique():
    for model_type_1 in df["Model Type"].unique():
        for model_type_2 in df["Model Type"].unique():
            if model_type_1 != model_type_2:
                per_1 = df[(df["Cross Validation Type"] == cv_type) & (df["Model Type"] == model_type_1)]
                per_2 = df[(df["Cross Validation Type"] == cv_type) & (df["Model Type"] == model_type_2)]

                gT, p = nm_stats.permutationTest_relative(
                    per_1["Test Performance"].values,
                    per_2["Test Performance"].values,
                    plot_distr=False,
                    x_unit=None,
                    p=5000
                )

                print(f"Cross Validation Type: {cv_type}, Model Type 1: {model_type_1}, Model Type 2: {model_type_2}")
                print(f"gT: {gT}, p-value: {p}")
                print("---------------------------------------------------")