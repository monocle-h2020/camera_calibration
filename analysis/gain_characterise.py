"""
Analyse gain maps (in ADU/electron) generated using the calibration functions.

Note: this script currently only looks at raw gain maps (ADU/electron at a
specific ISO speed), not normalised gain maps (normalised ADU/electron).

Command line arguments:
    * `file`: the location of the gain map to be analysed. This should be an
    NPY file generated using ../calibration/gain.py.
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, plot, symmetric_percentiles

# Get the data folder from the command line
file = io.path_from_input(argv)
root = io.find_root_folder(file)
ISO = io.split_iso(file)

# Load Camera object
camera = io.load_camera(root)
print(f"Loaded Camera object: {camera}")

# Save locations
savefolder = camera.filename_analysis("gain", makefolders=True)
save_to_histogram = savefolder/f"gain_histogram_iso{ISO}.pdf"
save_to_map = savefolder/f"gain_map_iso{ISO}.pdf"
save_to_histogram_miniature = savefolder/f"gain_histogram_iso{ISO}_rgb_only.pdf"

# Load the data
gains = np.load(file)
print("Loaded data")

# Plot an RGB histogram of the data
xmin, xmax = 0, symmetric_percentiles(gains, percent=0.001)[1]
camera.plot_histogram_RGB(gains, xmin=xmin, xmax=xmax, xlabel="Gain (ADU/e$^-$)", saveto=save_to_histogram)
print("Made histogram")

# Plot Gauss-convolved maps of the data
camera.plot_gauss_maps(gains, colorbar_label="Gain (ADU/e$^-$)", saveto=save_to_map)
print("Made maps")

# Demosaick data by splitting the RGBG2 channels into separate arrays
gains_RGBG = camera.demosaick(gains)

# Plot a miniature RGB histogram
xmin, xmax = 0, 3.5
data_RGB = [gains_RGBG[0].ravel(), gains_RGBG[1::2].ravel(), gains_RGBG[2].ravel()]
fig, axs = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(3.3,2.4), gridspec_kw={"wspace": 0, "hspace": 0})
for ax, data, c in zip(axs, data_RGB, plot.RGB_OkabeIto):
    ax.hist(data, color=c, edgecolor=c, bins=np.linspace(xmin, xmax, 250), density=True)
for ax in axs[:2]:
    ax.tick_params(bottom=False, labelbottom=False)
for ax in axs:
    ax.grid(ls="--")
axs[0].set_xlim(xmin, xmax)
axs[0].set_ylim(0, 2.5)
axs[0].set_yticks([0,1,2])
axs[2].set_xlabel("Gain (ADU/e-)")
axs[1].set_ylabel("Frequency")
plot._saveshow(save_to_histogram_miniature)
print("Made RGB histogram")
