import os
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

ind_peaks_long = {
    "rcs02l" : 20,
    "rcs02r" : 18,
    "rcs03l" : 13.5,
    "rcs05l" : 26,
    "rcs05r" : 26,
    "rcs06l" : 28,
    "rcs06r" : 18,
    "rcs07l" : 13,
    "rcs07r" : 8, # 
    "rcs08l" : 25,  # none
    "rcs08r" : 27,
    "rcs09l" : 24,
    "rcs09r" : 23,
    "rcs10l" : 27, # none
    "rcs10r" : 29,
    "rcs11l" : 27,
    "rcs11r" : 25,
    "rcs12l" : 28,
    "rcs12r" : 28, # none
    "rcs14l" : 25,
    "rcs15l" : 22,
    "rcs15r" : 18,
    "rcs17l" : 27,
    "rcs17r" : 29,
    "rcs18l" : 23, # none
    "rcs18r" : 23,
    "rcs19l" : 10,
    "rcs19r" : 22,
    "rcs20l" : 17,
    "rcs20r" : 17,
}

PATH_FIGURES = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf/figures_paper/figures_final'


df_ind_peaks = pd.DataFrame(ind_peaks_long.items(), columns=["sub", "peak_freq"])
df_ind_peaks["peak_freq"].mean()
df_ind_peaks["peak_freq"].std()
plt.figure(figsize=(3, 8.2/3))
sns.boxplot(data=df_ind_peaks, y="peak_freq", color="#31688E", showmeans=True, showfliers=False, boxprops=dict(alpha=0.5)
)
sns.swarmplot(data=df_ind_peaks, y="peak_freq", dodge=False, color="#31688E", alpha=0.9, s=5)
plt.ylabel("Frequency [Hz]")
plt.title("Individual peak frequency")
plt.savefig(os.path.join(PATH_FIGURES, "figure_ind_peaks.pdf"))
plt.show(block=True)
