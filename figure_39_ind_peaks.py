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

PATH_STIM = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/stimulation_settings.csv'
df_stim = pd.read_csv(PATH_STIM)
# convert the timestamp (in unix format) to datetime
df_stim["pkg_dt"] = pd.to_datetime(df_stim["timestamp"], unit="ms").dt.tz_localize("UTC")
df_stim.to_csv("stim_with_dt.csv")

df_all["pkg_dt"] = pd.to_datetime(df_all["pkg_dt"]).dt.tz_localize("US/Pacific").dt.tz_convert("UTC")

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
pdf_path = "ind_peaks_plots_daytime.pdf"

with PdfPages(pdf_path) as pdf:
    for sub in np.sort(subs):
        print(sub)
        #sub = "rcs12r"
        df_sub = df_all[df_all["sub"] == sub]
        df_sub = df_sub[(df_sub["h"] >= 8) & (df_sub["h"] <= 20)]
        df_psd = df_sub[[f'ch_subcortex_1_welch_psd_{i}_mean' for i in range(126)]]
        df_psd["pkg_dt"] = df_sub["pkg_dt"]
        # remove nan rows
        df_psd = df_psd.dropna().reset_index(drop=True)
        
        
        plt.figure(figsize=(15, 4)) 

        plt.subplot(122)
        # plot the ranges where stimulation_frequency > 0
        df_stim_patient = df_stim[np.logical_and(df_stim["subject_id"] == sub[:-1], df_stim["hemisphere_id"].apply(lambda x: x[0]) == sub[-1])].reset_index(drop=True)
        # sort by time
        df_stim_patient = df_stim_patient.sort_values("pkg_dt")
        stim_freq = df_stim_patient["stimulation_frequency"].values
        stim_ampl = df_stim_patient["stimulation_amplitude"].values
        stim_times_start = df_stim_patient["pkg_dt"][:-1].reset_index(drop=True)
        stim_times_stop = df_stim_patient["pkg_dt"][1:].reset_index(drop=True)
        idx_stim = []
        stim_ampl_arr = np.zeros(df_psd.shape[0])

        for i in range(len(stim_freq)-1):
            if stim_freq[i] > 0:
                # get range of stimulation
                x_start = stim_times_start[i]
                x_stop = stim_times_stop[i]
                idx_plt = np.where((df_psd["pkg_dt"] > x_start) & (df_psd["pkg_dt"] < x_stop))[0]
                
                if idx_plt.shape[0] > 0:
                    stim_ampl_arr[idx_plt] = stim_ampl[i]
                    #plt.axvspan(idx_plt[0], idx_plt[-1], facecolor="none", edgecolor="black", hatch="x", linewidth=0)
                    for j in range(idx_plt.shape[0]):
                        idx_stim.append(idx_plt[j])
                        plt.plot(np.arange(126), df_psd.iloc[idx_plt[j], :-1], alpha=0.1, color="red")
        idx_non_stim = np.setdiff1d(np.arange(df_psd.shape[0]), idx_stim)
        for i in idx_non_stim:
            plt.plot(np.arange(126), df_psd.iloc[i, :-1], alpha=0.05, color="blue")
        # highlight last stimulation
        if stim_freq[-1] > 0:
            x_start = stim_times_start.iloc[-1]
            idx_plt = np.where(df_psd["pkg_dt"] > x_start)[0]
            if idx_plt.shape[0] > 0:
                stim_ampl_arr[idx_plt] = stim_ampl[-1]
                #plt.axvspan(idx_plt[0], idx_plt[-1], facecolor="none", edgecolor="black", hatch="x", linewidth=0)
                for j in range(idx_plt.shape[0]):
                    plt.plot(np.arange(126), df_psd.iloc[idx_plt[j], :-1], alpha=0.1, color="red")

        #colors = sns.color_palette("cool", df_psd.shape[0])
        #for i in range(df_psd.shape[0]):
        #    plt.plot(np.arange(126), df_psd.iloc[i, :-1], alpha=0.05, color=colors[i])

        plt.xticks(np.arange(0, 126, 5))
        # delete every second tick label
        for label in plt.gca().get_xticklabels()[1::2]:
            label.set_visible(False)
        
        plt.xlabel("Frequency [Hz]")
        plt.ylabel("Power [dB]")
        plt.grid()

        plt.subplot(121)
        plt.imshow(df_psd.iloc[:, :-1].T, aspect="auto")
        plt.plot(np.arange(df_psd.shape[0]), stim_ampl_arr*10, color="white", linewidth=2, label="Stimulation amplitude")
        # flip y axis
        plt.gca().invert_yaxis()

        plt.suptitle(sub)
        


        # Save the current figure to the PDF
        pdf.savefig()
        plt.close()
