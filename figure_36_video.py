import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image

PATH_ = '/Users/Timon/Documents/py_neuro_ucsf/py_neuromodulation/video/ML_predictions_CatBoost Timon Merk.pkl'
d = pd.read_pickle(PATH_)

probas = d["all"]["ecog_stn"]["rcs02r"]["pr_proba"]

plt.figure(figsize=(15, 5))
plt.plot(d["all"]["ecog_stn"]["rcs02r"]["time"],
         d["all"]["ecog_stn"]["rcs02r"]["pr_proba"][:, 0])
plt.xlabel("Time")
plt.ylabel("Bradykinesia prediction")
plt.show(block=True)

time_ = d["all"]["ecog_stn"]["rcs02r"]["time"]
hours_ = pd.Series(time_).dt.hour
# make a barplot of hours sum
plt.figure()
hours_.value_counts().sort_index().plot(kind="bar")
plt.xlabel("Hour of the day")
plt.ylabel("Number of samples")
plt.title("Distribution of samples over the day")
plt.show(block=True)



step_size = 100
# plt.figure()
# plt.subplot(311)
# plt.plot(probas[:step_size,1])
# plt.title("Bradykinesia")
# # turn off x labels
# plt.xticks([])
# plt.subplot(312)
# plt.plot(probas[:step_size,0])
# plt.title("Dyskinesia")
# plt.xticks([])
# plt.subplot(313)
# plt.plot(probas[:step_size,2])
# plt.title("Tremor")
# plt.show(block=True)

from matplotlib.animation import FuncAnimation, PillowWriter

probas = probas[:, :]
# Sliding window size
window_size = 100

# Create figure and axes
fig = plt.figure(figsize=(10, 5))
gs = fig.add_gridspec(nrows=3, ncols=5)

# Image subplot (left column)
ax_image = fig.add_subplot(gs[:, :-1])  # rows, columns
ax_image.axis("off")  # Turn off axes for the image
image_display = ax_image.imshow(np.zeros((1671, 3266, 3), dtype=np.uint8))  # Placeholder for an RGB image
img_path = f"video/iter_1.png"  # 3266, 1671
img = Image.open(img_path)
image_display.set_data(img)
# Line plot subplots (right two columns)
ax_plots = [
    fig.add_subplot(gs[0, -1]),
    fig.add_subplot(gs[1, -1]),
    fig.add_subplot(gs[2, -1])  # Span the bottom row across two columns
]

# Titles for subplots
titles = ["Bradykinesia", "Dyskinesia", "Tremor"]
lines = []

for ax, title in zip(ax_plots, titles):
    ax.set_xlim(0, window_size - 1)  # Fixed x-axis for the sliding window
    ax.set_ylim(0, 1)               # y-axis (assuming probabilities range from 0 to 1)
    ax.set_title(title)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    # set x ticks to 5 ticks
    ax.set_xticks(np.arange(0, window_size, 30))
    ax.set_xticklabels(np.array(np.arange(0, window_size, 30) / 30, dtype=int))
    # set xlabel to "Time [h]"
    ax.set_xlabel("Time [h]")
    # turn x labels and xticklabels if not last subplot
    if ax != ax_plots[-1]:
        ax.set_xticklabels([])
        ax.set_xlabel("")
        ax.set_xticks([])
    if title == "Dyskinesia":
        ax.set_ylabel(f"Prediction probability")

    line, = ax.plot([], [], color='grey')  # Initialize an empty line plot with grey color
    lines.append(line)

plt.tight_layout()
#plt.show(block=True)
# Update function for animation
def update(frame):
    start = max(0, frame - window_size + 1)
    end = frame + 1
    
    for i, line in enumerate(lines):
        line.set_data(range(end - start), probas[start:end, i])
    
    if frame % 8 == 0:
        # update image
        img_idx = frame // 8 + 1
        img_path = f"video/iter_{img_idx}.png"
        img = Image.open(img_path)
        image_display.set_data(img)
    

    return lines + [image_display]

# Create animation
n_frames = len(probas)  # Total number of frames
ani = FuncAnimation(fig, update, frames=n_frames, blit=True, interval=50)  # Adjust interval for frame rate

# Save animation as video (MP4)
ani.save("sliding_window_animation.mp4", writer="ffmpeg", fps=20)

# Optional: Save animation as GIF
# ani.save("sliding_window_animation.gif", writer=PillowWriter(fps=20))

plt.show()