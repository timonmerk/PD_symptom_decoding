import pandas as pd
import os
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.cm as cm


PATH_FEATURES = "/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_normalized_10s_window_length/480"
PATH_FIGURES = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf'
df_all = pd.read_csv(os.path.join(PATH_FEATURES, "all_merged_normed.csv"), index_col=0)
df_all = pd.read_csv('/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/merged_rmap/all_ch_renamed_no_rmap.csv')
df_all["pkg_dt"] = pd.to_datetime(df_all["pkg_dt"]).dt.tz_localize("US/Pacific")
subs = np.sort(df_all["sub"].unique())

df_stim_times = pd.read_csv('/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per/stim_ampl_freq.csv')
df_stim_times["pkg_dt"] = pd.to_datetime(df_stim_times["pkg_dt"], utc=True).dt.tz_convert("US/Pacific")
#OUT_FILE = "LOHO_ALL_LABELS_ALL_GROUPS_normed_480.pkl"

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

df_all["h"] = df_all["pkg_dt"].dt.hour
# Plot example prediction beta vs decoding

LIMIT_FIRST_15_SAMPLES = False

if LIMIT_FIRST_15_SAMPLES:  
    pdf_path = "figure_40_ind_peaks_plots_daytime_first30min.pdf"
    alpha=0.6
else:
    pdf_path = "figure_40_ind_peaks_plots_daytime.pdf"
    alpha=0.1
pdf_path = os.path.join(PATH_FIGURES, pdf_path)
mean_spectra_all = []

with PdfPages(pdf_path) as pdf:
    for sub in np.sort(subs):
        mean_spectra_sub = []
        print(sub)
        #sub = "rcs05l"
        df_sub = df_all[df_all["sub"] == sub]
        df_stim_sub = df_stim_times[df_stim_times["sub"] == sub]
        df_stim_sub = df_stim_sub.sort_values("pkg_dt")
        # merge the stimulation times with the features
        df_sub = pd.merge(df_sub, df_stim_sub, on="pkg_dt", how="left")
        
        df_sub = df_sub[(df_sub["h"] >= 8) & (df_sub["h"] <= 20)]

        df_psd = df_sub[[f'ch_subcortex_1_welch_psd_{i}_mean' for i in range(126)]]
        df_psd["pkg_dt"] = df_sub["pkg_dt"]
        df_psd["stim_freq"] = df_sub["stim_freq"]
        df_psd["stim_ampl"] = df_sub["stim_ampl"]
        # remove nan rows
        df_psd = df_psd.dropna().reset_index(drop=True)
        
        plt.figure(figsize=(15, 4)) 

        plt.subplot(122)
        
        df_stim = df_psd[df_psd["stim_freq"] > 0]
        df_stimoff = df_psd[df_psd["stim_freq"] == 0]
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
            plt.plot(np.arange(15), df_stimoff["stim_ampl"].iloc[:15].values*10, color="white", linewidth=2, label="Stimulation amplitude")
        else:
            plt.imshow(df_psd.iloc[:, :-3].T, aspect="auto")
            plt.plot(np.arange(df_psd.shape[0]), df_psd["stim_ampl"].values*10, color="white", linewidth=2, label="Stimulation amplitude")

        plt.legend()
        plt.gca().invert_yaxis()
        plt.suptitle(sub)
        pdf.savefig()
        plt.close()
    
        mean_spectra_all.append(np.array(mean_spectra_sub).mean(axis=0))

mean_spectra = np.array(mean_spectra_all)

plt.figure(figsize=(8, 8))
n = mean_spectra.shape[0]
# Get viridis colormap with n evenly spaced colors
viridis = cm.get_cmap('jet', n)
colors = [viridis(i) for i in range(n)]

for i in range(mean_spectra.shape[0]):
    plt.plot(np.arange(126), mean_spectra[i, :], alpha=1, label=subs[i], color=colors[i])

plt.xlabel("Frequency [Hz]")
plt.ylabel("Power [dB]")
plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.grid()
plt.xticks(np.arange(0, 126, 5))
plt.tight_layout()

if LIMIT_FIRST_15_SAMPLES:
    plt.title("Mean spectra of first 15 samples without stimulation")
    plt.savefig(os.path.join(PATH_FIGURES, "figure_40_mean_spectra_first15.pdf"))
else:
    plt.title("Mean spectra without stimulation")
    plt.savefig(os.path.join(PATH_FIGURES, "figure_40_mean_spectra.pdf"))
plt.show(block=True)



