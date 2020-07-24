"""
Plot Bayer RGBG2 maps of any number of data sets into a single plot. The plot
has four columns (one per channel) and N rows, with N the number of data
files provided. All subplots share a common colorbar (in red/green/blue).

Command line arguments:
    * `files`: NPY stacks to be plotted.
    (multiple arguments possible)
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import plot, io, analyse
from spectacle.general import gaussMd

# Get the data folder from the command line
files = io.path_from_input(argv)
roots = [io.find_root_folder(path) for path in files]

# Future command line arguments
save_to = io.results_folder/"RGBG.pdf"
colorbar_label = 40*" " + "Read noise (ADU)"

# Load Camera object
cameras = [io.load_camera(root) for root in roots]
print(f"Loaded Camera objects: {cameras}")

# Load the data
data_all = [np.load(path) for path in files]

# Demosaick the data
RGBGs_all = [camera.demosaick(data) for data, camera in zip(data_all, cameras)]

# Convolve the data with a Gaussian kernel
gauss_all = [gaussMd(RGBG, sigma=(0,5,5)) for RGBG in RGBGs_all]

# Minimum and maximum of colourbars
vmin, vmax = analyse.symmetric_percentiles(gauss_all)

# Figure to hold the subplots
fig, axs = plt.subplots(ncols=4, nrows=len(files), sharex=True, sharey=True, figsize=(7, 1.695*len(files)), squeeze=False, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0})

# Loop over the data and plot them
for path, gauss, axs_here in zip(files, gauss_all, axs):

    # Loop over the Bayer RGBG2 channels
    for j, (ax, D, c) in enumerate(zip(axs_here, gauss, plot.RGBG2)):

        # Plot the image
        img = ax.imshow(D, cmap=plot.cmaps[c+"r"], vmin=vmin, vmax=vmax)

        # Remove the x- and y-axes
        ax.set_xticks([])
        ax.set_yticks([])

        # For the lowest row only, add a colorbar
        if ax is axs[-1,j]:
            colorbar_here = plot.colorbar(img)
            if ax is axs_here[1]:
                colorbar_here.set_label(colorbar_label)
            colorbar_here.locator = plot.ticker.MaxNLocator(nbins=3)
            colorbar_here.update_ticks()

# Save the figure
fig.savefig(save_to)
plt.close()
print(f"Saved figure to '{save_to}'")
