import pandas as pd
import os
import pickle
from matplotlib import pyplot as plt
import numpy as np
from scipy import stats
import seaborn as sns

PATH_PER = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/out_per'
OUT_FILE = "LOHO_ALL_LABELS_ALL_GROUPS_normed_480.pkl"
PATH_FIGURES = '/Users/Timon/Library/CloudStorage/OneDrive-Charité-UniversitätsmedizinBerlin/Shared Documents - ICN Data World/General/Data/UCSF_OLARU/figures_ucsf'

PATH_READ = os.path.join(PATH_PER, OUT_FILE)

with open(PATH_READ, "rb") as f:
    d_out = pickle.load(f)

times_all = []
for sub in d_out[False]["pkg_dk"]["ecog_stn"].keys():
    df_ = pd.DataFrame()
    df_["time"] = d_out[False]["pkg_dk"]["ecog_stn"][sub]["time"]
    df_["sub"] = sub
    times_all.append(df_)
times_all = pd.concat(times_all, axis=0)
times_all["hour"] = pd.to_datetime(times_all["time"]).dt.hour

# make a seaborn histplot of the hour, colorcode subjects
plt.figure(figsize=(5, 10))
ax = sns.histplot(data=times_all, x="hour", hue="sub", multiple="stack", bins=24, palette="viridis")
sns.move_legend(ax, "upper left", bbox_to_anchor=(1, 1))
plt.xlabel("Hour of the day")
plt.ylabel("Number of samples")
plt.title("Distribution of samples over the day")
plt.tight_layout()
plt.show(block=True)


times_all["radian"] = times_all["hour"] * (2 * np.pi / 24)

# Compute histogram data
bins = np.linspace(0, 2 * np.pi, 24)  # 24 bins, covering the full circle
hist_data = times_all.groupby(["sub", pd.cut(times_all["radian"], bins)]).size().unstack(fill_value=0)

# Create polar plot
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='polar')

# Define colors for each subject
colors = plt.cm.jet(np.linspace(0, 1, len(hist_data)))

# Plot histogram as stacked bars
bottom = np.zeros(len(bins) - 1)
for i, (sub, counts) in enumerate(hist_data.iterrows()):
    ax.bar(bins[:-1], counts, width=(2 * np.pi / 24), bottom=bottom, color=colors[i], alpha=0.75, label=sub)
    bottom += counts  # Stack the bars

# Adjust aesthetics
ax.set_theta_zero_location("N")  # Start at 12 o'clock
ax.set_theta_direction(-1)  # Clockwise direction
ax.set_xticks(np.linspace(0, 2 * np.pi, 8, endpoint=False))  # Set major ticks
ax.set_xticklabels(["00:00", "03:00", "06:00", "09:00", "12:00", "15:00", "18:00", "21:00"])  # Labels

# Add title and legend
plt.title("Distribution of samples over the day", pad=20)
plt.legend(loc="upper right", bbox_to_anchor=(1.2, 1.2))
plt.tight_layout()
# Show the plot
plt.savefig(os.path.join(PATH_FIGURES, "figure_38_time_hist_polar.pdf"))
plt.show(block=True)

