import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy import stats
from sklearn.preprocessing import MinMaxScaler

# === Global font ===
plt.rcParams["font.family"] = "Arial"

# === Load data ===
df = pd.read_csv("connectomic_Patricia/df_rcs02_left.csv")

# === Normalize Dyskinesia Decoder ===
scaler = MinMaxScaler()

# === Build data dictionary ===
data = {
    "Bradykinesia": np.column_stack([
        stats.zscore(df["Timon_B"].values),
        stats.zscore(df["CT_B"].values)
    ]),
    "Tremor": np.column_stack([
        stats.zscore(df["Timon_T"].values),
        stats.zscore(df["CT_T"].values)
    ]),
    "Dyskinesia": np.column_stack([
        scaler.fit_transform(df[["Timon_D"]].values).flatten(),  # normalized 0–1
        df["CT_Amplitude"].values                                # raw amplitude
    ])
}

# === Parameters ===
window_size = 30
fps = 6
interval = 1000 / fps
n_frames = len(df)
fontsize = 20

# === Figure ===
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
titles = ["Bradykinesia", "Tremor", "Dyskinesia"]
colors = [
    ["#A1C2EC", "#376EB4"],   # Brady
    ["#A1ECCC", "#29AF7F"],   # Tremor
    ["#F5BFBF", "#DF4A4A"]    # Dyskinesia
]
lines = []

# === Compute y-limits for z-scored ===
z_values = np.concatenate([data["Bradykinesia"], data["Tremor"]])
zmin, zmax = np.nanmin(z_values), np.nanmax(z_values)
pad = 0.05 * (zmax - zmin)
zmin -= pad
zmax += pad

# --- Plot setup ---
for i, (ax, title, cols) in enumerate(zip(axes, titles, colors)):
    ax.set_title(title, fontsize=fontsize + 2)
    ax.set_xlim(0, window_size)

    if title in ["Bradykinesia", "Tremor"]:
        # === z-scored plots ===
        ax.set_ylim(zmin, zmax)
        ax.set_ylabel("Decoder /\nTracts Activated", fontsize=15)
        line1, = ax.plot([], [], color=cols[0], lw=2, label="Decoder")
        line2, = ax.plot([], [], color=cols[1], lw=2, label="Cleartune")
        lines.append((line1, line2))
        ax.legend(frameon=False, fontsize=fontsize - 4, loc="upper right")

        # Remove top & right spines
        for spine in ["top", "right"]:
            ax.spines[spine].set_visible(False)

    else:
        # === Dyskinesia subplot (normalized + raw amplitude) ===
        ax.set_ylim(0, 1)
        ax.set_ylabel("Dyskinesia Decoder", fontsize=15)
        ax2 = ax.twinx()
        ax2.set_ylim(2.3, 3.6)  # fixed amplitude range
        ax2.set_ylabel("Amplitude [mA]", fontsize=15)

        line1, = ax.plot([], [], color=cols[0], lw=2, label="Decoder")
        line2, = ax2.plot([], [], color=cols[1], lw=2, label="Cleartune")
        lines.append((line1, line2))

        ax.legend(frameon=False, fontsize=fontsize - 4, loc="upper left")
        ax2.legend(frameon=False, fontsize=fontsize - 4, loc="upper right")

        # === Custom x-axis ticks and label ===
        ax.set_xticks(np.linspace(0, window_size, 7))
        ax.set_xticklabels([0, 1, 2, 3, 4, 5, 6])
        ax.set_xlabel("Time [h]", fontsize=fontsize)

        # Remove only top spine (keep right spine)
        ax.spines["top"].set_visible(False)
        ax2.spines["top"].set_visible(False)

    ax.tick_params(labelsize=fontsize - 4)

plt.tight_layout()

# === Update function ===
def update(frame):
    start = max(0, frame - window_size)
    end = frame
    x = np.arange(end - start)
    for (line1, line2), key in zip(lines, data.keys()):
        y = data[key]
        line1.set_data(x, y[start:end, 0])  # Decoder
        line2.set_data(x, y[start:end, 1])  # Cleartune
    return [l for pair in lines for l in pair]

# === Animation ===
ani = FuncAnimation(fig, update, frames=n_frames, blit=True, interval=interval)

# === Save ===
writer = FFMpegWriter(fps=fps, codec="libx264")
ani.save("time_traces_final_Arial_fps6_window30_timeAxis.mp4", writer=writer)
plt.show()