import pickle
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PATH_FIGURES = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf'
PATH_OUT = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per'
with open(os.path.join(PATH_OUT, "stim_times_subs.pkl"), "rb") as f:
    stim_times_sub = pickle.load(f)

with open(os.path.join(PATH_OUT, "artifact_times_subs.pkl"), "rb") as f:
    artifact_times_sub = pickle.load(f)

df_all = pd.read_csv('/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_rmap/all_ch_renamed_no_rmap.csv')
df_all["pkg_dt"] = pd.to_datetime(df_all["pkg_dt"]).dt.tz_localize("US/Pacific")
subs = np.sort(df_all["sub"].unique())
pdf_path = os.path.join(PATH_FIGURES, "man_stim_stim_off.pdf")

LIMIT_FIRST_15_SAMPLES = False

if LIMIT_FIRST_15_SAMPLES:
    alpha = 0.5
else:
    alpha=0.2

with PdfPages(pdf_path) as pdf:
    for sub in np.sort(subs):
        mean_spectra_sub = []
        print(sub)
        #sub = "rcs17l"
        df_sub = df_all[df_all["sub"] == sub]

        df_sub["stim"] = False
        df_sub["artifact"] = False
        
        if sub in stim_times_sub.keys():
            for time in stim_times_sub[sub]:
                df_ts = df_sub[df_sub["pkg_dt"] == time]
                if df_ts.shape[0] > 0:
                    idx = df_ts.index[0]
                    df_sub.at[idx, "stim"] = True
        if sub in artifact_times_sub.keys():
            for time in artifact_times_sub[sub]:
                df_ts = df_sub[df_sub["pkg_dt"] == time]
                if df_ts.shape[0] > 0:
                    idx = df_ts.index[0]
                    df_sub.at[idx, "artifact"] = True
        
        #df_sub = df_sub[(df_sub["h"] >= 8) & (df_sub["h"] <= 20)]

        df_psd = df_sub[[f'ch_subcortex_1_welch_psd_{i}_mean' for i in range(126)]]
        df_psd["pkg_dt"] = df_sub["pkg_dt"]
        df_psd["stim"] = df_sub["stim"]
        df_psd["artifact"] = df_sub["artifact"]
        # remove nan rows
        df_psd = df_psd.dropna().reset_index(drop=True)
        
        plt.figure(figsize=(15, 8)) 

        plt.subplot(122)
        
        df_stim = df_psd[df_psd["stim"] == True]
        df_stimoff = df_psd[df_psd["stim"] == False]
        df_artifact = df_psd[df_psd["artifact"] == True]

        if LIMIT_FIRST_15_SAMPLES is False:
            for cnt, idx in enumerate(df_stim.index):
                plt.plot(np.arange(126), df_psd.iloc[idx, :-3], color="red", alpha=alpha)
        for cnt, idx in enumerate(df_stimoff.index):
            if LIMIT_FIRST_15_SAMPLES:
                if cnt < 15:
                    mean_spectra_sub.append(df_psd.iloc[idx, :-3])
                    plt.plot(np.arange(126), df_psd.iloc[idx, :-3], color="blue", alpha=alpha)
            else:
                plt.plot(np.arange(126), df_psd.iloc[idx, :-3], color="blue", alpha=alpha)
                mean_spectra_sub.append(df_psd.iloc[idx, :-3])
        for cnt, idx in enumerate(df_artifact.index):
            plt.plot(np.arange(126), df_psd.iloc[idx, :-3], color="purple", alpha=alpha)

        plt.xticks(np.arange(0, 126, 5))
        # delete every second tick label
        for label in plt.gca().get_xticklabels()[1::2]:
            label.set_visible(False)
        
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power [dB]")
        plt.grid()

        plt.subplot(121)
        if LIMIT_FIRST_15_SAMPLES:
            plt.imshow(df_stimoff.iloc[:15, :-3].T, aspect="auto")
            plt.plot(np.arange(15), df_stimoff["stim"].iloc[:15].values*10, color="white", linewidth=2, label="Stimulation amplitude")
        else:
            plt.imshow(df_psd.iloc[:, :-4].T, aspect="auto")
            plt.plot(np.arange(df_psd.shape[0]), df_psd["stim"].values*10, color="white", linewidth=2, label="Stimulation amplitude")
            # plot also hour of day
            #plt.plot(np.arange(df_psd.shape[0]), df_psd["h"].values, color="black", linewidth=2, label="Hour of day")
        #plt.legend()
        plt.gca().invert_yaxis()
        plt.suptitle(sub)
        pdf.savefig()
        #if sub == "rcs12r":
        #plt.show(block=True)
        plt.close()
    