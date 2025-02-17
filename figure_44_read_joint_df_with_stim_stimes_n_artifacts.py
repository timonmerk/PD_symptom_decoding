import pickle
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PATH_FIGURES = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf'
PATH_OUT = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per'

df_all = pd.read_csv('/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/all_artifact_and_stim_condition.csv')
df_all["pkg_dt"] = pd.to_datetime(df_all["pkg_dt"], utc=True).dt.tz_convert("US/Pacific")
subs = np.sort(df_all["sub"].unique())
pdf_path = os.path.join(PATH_FIGURES, "man_stim_stim_off_check2.pdf")

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
        
        #df_sub = df_sub[(df_sub["h"] >= 8) & (df_sub["h"] <= 20)]

        df_psd = df_sub[[f'ch_subcortex_1_welch_psd_{i}_mean' for i in range(126)]]
        df_psd["pkg_dt"] = df_sub["pkg_dt"]
        df_psd["condition"] = df_sub["condition"]
        # remove nan rows
        df_psd = df_psd.dropna().reset_index(drop=True)
        
        plt.figure(figsize=(15, 8)) 

        plt.subplot(122)
        
        df_stim = df_psd[df_psd["condition"] == "stim"]
        df_stimoff = df_psd[df_psd["condition"] == "stim_off"]
        df_artifact = df_psd[df_psd["condition"] == "artifact"]

        if LIMIT_FIRST_15_SAMPLES is False:
            for cnt, idx in enumerate(df_stim.index):
                plt.plot(np.arange(126), df_psd.iloc[idx, :-2], color="red", alpha=alpha)
        for cnt, idx in enumerate(df_stimoff.index):
            if LIMIT_FIRST_15_SAMPLES:
                if cnt < 15:
                    mean_spectra_sub.append(df_psd.iloc[idx, :-2])
                    plt.plot(np.arange(126), df_psd.iloc[idx, :-2], color="blue", alpha=alpha)
            else:
                plt.plot(np.arange(126), df_psd.iloc[idx, :-2], color="blue", alpha=alpha)
                mean_spectra_sub.append(df_psd.iloc[idx, :-2])
        for cnt, idx in enumerate(df_artifact.index):
            plt.plot(np.arange(126), df_psd.iloc[idx, :-2], color="black", alpha=alpha)

        plt.xticks(np.arange(0, 126, 5))
        # delete every second tick label
        for label in plt.gca().get_xticklabels()[1::2]:
            label.set_visible(False)
        
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power [dB]")
        plt.grid()

        plt.subplot(121)
        if LIMIT_FIRST_15_SAMPLES:
            plt.imshow(df_stimoff.iloc[:15, :-2].T, aspect="auto")
            plt.plot(np.arange(15), df_stimoff["stim"].iloc[:15].values*10, color="white", linewidth=2, label="Stimulation amplitude")
        else:
            plt.imshow(df_psd.iloc[:, :-2].T, aspect="auto")
            #plt.plot(np.arange(df_psd.shape[0]), df_psd["stim"].values*10, color="white", linewidth=2, label="Stimulation amplitude")
            # plot also hour of day
            #plt.plot(np.arange(df_psd.shape[0]), df_psd["h"].values, color="black", linewidth=2, label="Hour of day")
        #plt.legend()
        plt.gca().invert_yaxis()
        plt.suptitle(sub)
        pdf.savefig()
        #if sub == "rcs12r":
        #plt.show(block=True)
        plt.close()
    