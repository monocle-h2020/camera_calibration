"""
Analyse gain maps (in ADU/electron) generated using the calibration functions.

Note: this script currently only looks at raw gain maps (ADU/electron at a
specific ISO speed), not normalised gain maps (normalised ADU/electron).

Command line arguments:
    * `file`: the location of the gain map to be analysed.
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import raw, io, plot, analyse
from spectacle.general import gauss_nan

# Get the data folder from the command line
file = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(file)
savefolder = root/"results/gain"
ISO = io.split_iso(file)

# Get metadata
camera = io.load_metadata(root)
print("Loaded metadata")

# Load the data
gains = np.load(file)
print("Loaded data")

# Demosaick data by splitting the RGBG2 channels into separate arrays
gains_RGBG,_ = raw.pull_apart(gains, camera.bayer_map)

# Convolve the data with a Gaussian kernel
gains_combined_gauss = gauss_nan(gains, sigma=10)
gains_gauss = gauss_nan(gains_RGBG, sigma=(0,5,5))

# Plot an RGB histogram of the data
xmin, xmax = 0, analyse.symmetric_percentiles(gains, percent=0.001)[1]
analyse.plot_histogram_RGB(gains, camera.bayer_map, xlim=(xmin, xmax), xlabel="Gain (ADU/e$^-$)", saveto=savefolder/f"gain_histogram_iso{ISO}.pdf")
print("Made histogram")

# Plot Gauss-convolved maps of the data
# Note: cannot use analyse.plot_gauss_maps because of NaNs
plot.show_image(gains_combined_gauss, colorbar_label="Gain (ADU/e$^-$)", saveto=savefolder/f"gain_map_iso{ISO}.pdf")
plot.show_image_RGBG2(gains_gauss, colorbar_label="Gain (ADU/e$^-$)", saveto=savefolder/f"gain_map_iso{ISO}.pdf")
print("Made maps")

# Plot a miniature RGB histogram
fig, axs = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(3.3,2.4), squeeze=True, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0})
axs[0].hist(gains_RGBG[0]   .ravel(), bins=np.linspace(0, 3.5, 250), color="r", edgecolor="r", density=True)
axs[1].hist(gains_RGBG[1::2].ravel(), bins=np.linspace(0, 3.5, 250), color="g", edgecolor="g", density=True)
axs[2].hist(gains_RGBG[2]   .ravel(), bins=np.linspace(0, 3.5, 250), color="b", edgecolor="b", density=True)
for ax in axs[:2]:
    ax.xaxis.set_ticks_position("none")
for ax in axs:
    ax.grid(True)
    ax.set_yticks([0,1,2])
axs[0].set_xlim(0, 3.5)
axs[0].set_ylim(0, 2.5)
axs[2].set_xlabel("Gain (ADU/e-)")
axs[1].set_ylabel("Frequency")
plt.savefig(savefolder/f"gain_histogram_iso{ISO}_rgb_only.pdf")
plt.close()
print("Made RGB histogram")
