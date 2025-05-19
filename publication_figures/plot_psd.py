import pickle
import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

PATH_FIGURES = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf/figures_paper/figures_final'
PATH_OUT = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per'

df_all = pd.read_csv('/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/features/all_artifact_and_stim_condition.csv')
df_all["pkg_dt"] = pd.to_datetime(df_all["pkg_dt"], utc=True).dt.tz_convert("US/Pacific")
df_all["h"] = df_all["pkg_dt"].dt.hour
df_all = df_all.query("h >= 8 and h <= 20")
subs = np.sort(df_all["sub"].unique())
patients_without_peak = ["rcs08l", "rcs10l", "rcs12r", "rcs18l"]  #  "rcs07r",

LIMIT_FIRST_15_SAMPLES = False

if LIMIT_FIRST_15_SAMPLES:
    alpha = 0.5
else:
    alpha=0.2

#pdf_pages = PdfPages(os.path.join(PATH_FIGURES, "psds.pdf"))
plt.figure(figsize=(3, 5))

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
    
    #df_stim = df_psd[df_psd["condition"] == "stim"]
    df_stimoff = df_psd[df_psd["condition"] == "stim_off"]
    #df_artifact = df_psd[df_psd["condition"] == "artifact"]
    if LIMIT_FIRST_15_SAMPLES:
        df_stimoff = df_stimoff.iloc[:15, :]
        #df_artifact = df_artifact.iloc[:15, :]
    mean_psd_sub = df_stimoff.iloc[:, :-2].mean().values

    color = "black" if sub not in patients_without_peak else "red"
    #if sub in patients_without_peak:
    #    plt.plot(np.arange(126), mean_psd_sub, alpha=1, label=sub)  # color=color,
    #plt.figure(figsize=(3, 5))
    plt.grid()
    plt.plot(np.arange(126), mean_psd_sub, alpha=1, label=sub, color=color)  # color=color,
    plt.xticks(np.arange(0, 126, 5))
    
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Power [dB]")
    # plt.grid()
    plt.xlim(0, 50)
    #plt.legend()
    plt.ylim(-14, -9)
    # plot vertical lines at 8 and 30 
    plt.axvline(x=8, color='gray', linestyle='--', alpha=alpha)
    plt.axvline(x=30, color='gray', linestyle='--', alpha=alpha)
    #plt.title(sub)
    # show the grid 
    #pdf_pages.savefig(plt.gcf(), bbox_inches='tight', dpi=300)
    #plt.close()
#pdf_pages.close()
plt.savefig(os.path.join(PATH_FIGURES, "psds_all.pdf"), bbox_inches='tight', dpi=300)

plt.show()
