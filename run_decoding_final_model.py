import os
import pickle
import pandas as pd
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from sklearn import linear_model
from catboost import CatBoostRegressor, CatBoostClassifier
import skops.io as sio

if __name__ == "__main__":
    PATH_READ = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_std_10s_window_length"
    PATH_OUT = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/final_models"
    INCLUDE_PSD = True
    
    df_orig = pd.read_csv(os.path.join(PATH_READ, "all_merged_preprocessed_with_condition_pkgnormed.csv"), index_col=0)
    df_orig = df_orig[df_orig["condition"] == "stim_off"]
    df_orig = df_orig.drop(columns=["condition"])

    d_out_feature_importances = {}
    for MODEL_NAME in ["CB", "LM"]:
        if MODEL_NAME == "LM":
            # save feature importances to pickle
            with open(os.path.join(PATH_OUT, f"feature_importances_cb.pkl"), "wb") as f:
                pickle.dump(d_out_feature_importances, f)
        for CLASSIFICATION in [False, ]:
            for label_name in ["pkg_dk", "pkg_tremor", "pkg_bk"]:
                for loc_ in ["ecog", "stn", "ecog_stn"]:
                    
                    df_all = df_orig.copy()
                    df_all = df_all.drop(columns=df_all.columns[df_all.isnull().all()])
                    df_all["pkg_dt"] = pd.to_datetime(df_all["pkg_dt"], utc=True).dt.tz_convert("US/Pacific")
                    df_all = df_all[(df_all["pkg_dt"].dt.hour >= 8) & (df_all["pkg_dt"].dt.hour <= 20)]  # exclude night time recordings

                    mask = ~df_all[label_name].isnull()
                    df_all = df_all[mask]
                    
                    if loc_ == "ecog_stn":
                        df_use = df_all.copy()
                    elif loc_ == "ecog":
                        df_use = df_all[[c for c in df_all.columns if c.startswith("ch_cortex") or c.startswith("pkg") or c.startswith("sub")]].copy()
                    elif loc_ == "stn":
                        df_use = df_all[[c for c in df_all.columns if c.startswith("ch_subcortex") or c.startswith("pkg") or c.startswith("sub")]].copy()
                    if CLASSIFICATION:
                        if "_dk" in label_name:
                            df_use[label_name] = (df_use[label_name].copy() / df_use[label_name].max()) > 0.02
                        elif "_tremor" in label_name:
                            df_use[label_name] = df_use[label_name].copy() > 1
                        elif "_bk" in label_name:
                            df_use[label_name] = df_use[label_name].copy() > 50

                    df_train = df_use
                    df_train = df_train.drop(columns=["sub"])
                    y_train = np.array(df_train[label_name])

                    cols_use = [c for c in df_train.columns if "pkg" not in c]
                    if not INCLUDE_PSD:
                        cols_use = [c for c in cols_use if "psd" not in c]
                    
                    X_train = df_train[
                        cols_use
                    ]
                    X_train["hour"] = df_train["pkg_dt"].dt.hour
                
                    if CLASSIFICATION:
                        classes = np.unique(y_train)
                        weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
                        class_weights = dict(zip(classes, weights))
                        if MODEL_NAME == "CB":
                            model = CatBoostClassifier(
                                silent=True,
                                class_weights=class_weights,
                            )
                        elif MODEL_NAME == "LM":
                            model = linear_model.LogisticRegression(class_weight="balanced")
                    else:
                        if MODEL_NAME == "CB":
                            model = CatBoostRegressor(silent=True)
                        elif MODEL_NAME == "LM":
                            model = linear_model.LinearRegression()

                    if MODEL_NAME != "CB":
                        X_train = X_train.dropna(axis=1)
                        X_train = X_train.replace([np.inf, -np.inf], np.nan)
                        X_train = X_train.dropna(axis=1)

                    model.fit(X_train, y_train)

                    if MODEL_NAME == "CB":
                        if label_name not in d_out_feature_importances:
                            d_out_feature_importances[label_name] = {}
                        d_out_feature_importances[label_name][loc_] = {}
                        d_out_feature_importances[label_name][loc_]["feature_importances"] = model.feature_importances_
                        d_out_feature_importances[label_name][loc_]["feature_names"] = X_train.columns.tolist()

                    # save the trained model to d_out
                    model_name = f"{MODEL_NAME}_{'classification' if CLASSIFICATION else 'regression'}_{label_name[4:]}_{loc_}"
                    if MODEL_NAME == "LM":
                        model_name = model_name + ".skops"
                        model_skops = sio.dump(model, os.path.join(PATH_OUT, model_name))
                    else:
                        model_name = model_name + ".cbm"
                        model.save_model(os.path.join(PATH_OUT, model_name))

