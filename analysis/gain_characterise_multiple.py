"""
Analyse gain maps (in ADU/electron) generated using the calibration functions.
Multiple maps (for example different cameras or different ISO speeds) are
plotted at once.

Note: this script currently only looks at raw gain maps (ADU/electron at a
specific ISO speed), not normalised gain maps (normalised ADU/electron).

Command line arguments:
    * `file`: the location of the gain map to be analysed. This should be an
    NPY file generated using ../calibration/gain.py.
    (multiple arguments possible)
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, plot, analyse
from spectacle.general import gauss_filter_multidimensional

# Get the data folder from the command line
files = io.path_from_input(argv)
roots = [io.find_root_folder(file) for file in files]
save_folder = io.results_folder

# Load Camera objects
cameras = [io.load_camera(root) for root in roots]
print(f"Loaded Camera objects: {cameras}")

# Find the ISO speed for each gain map, to include in the plot titles
isos = [io.split_iso(file) for file in files]

# Labels for the plots, based on camera and ISO
labels = [f"{camera.name} (ISO {iso})" for camera, iso in zip(cameras, isos)]

# Load the data
data_arrays = [np.load(file) for file in files]
print("Loaded data")

# Demosaick the data (split into the Bayer RGBG2 channels)
data_RGBG_arrays = [camera.demosaick(data) for data, camera in zip(data_arrays, cameras)]
print("Demosaicked data")

# Convolve the RGBG2 data with a Gaussian kernel
data_RGBG_gauss_arrays = [gauss_filter_multidimensional(RGBG, sigma=(0,5,5)) for RGBG in data_RGBG_arrays]
print("Gaussed data")

# Plot a Gaussed map for each channel
# Loop over the Bayer RGBG2 channels
for j, c in enumerate(plot.rgbg2):
    # Create a figure to plot into
    fig, axs = plt.subplots(ncols=len(files), figsize=(3*len(files), 2.3), squeeze=True, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0})
    # Loop over the demosaicked/gaussed gain maps and plot them into the figure
    for label, ax, data_RGBG, data_RGBG_gauss in zip(labels, axs, data_RGBG_arrays, data_RGBG_gauss_arrays):
        # Uppercase label for colour
        c_label = c.upper()

        # Plot channel j into this figure
        im = ax.imshow(data_RGBG_gauss[j], cmap=plot.cmaps[c+"r"])

        # Plot parameters
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(label)

        # Include a colorbar
        # Left-most map has a colorbar on the left
        if ax is axs[0]:
            loc = "left"
        # Right-most map has a colorbar on the right
        elif ax is axs[-1]:
            loc = "right"
        # Any other maps have a colorbar on the bottom
        else:
            loc = "bottom"
        cbar = plot.colorbar(im, location=loc, label="Gain (ADU/e$^-$)")

        # Print the range of gain values found in this map
        percentile_low, percentile_high = analyse.symmetric_percentiles(data_RGBG)
        print(label)
        print(f"{c_label:>2}: {percentile_low:.2f} -- {percentile_high:.2f}")

    # Save the figure
    save_to_map_c = save_folder/f"gain_map_{c_label}.pdf"
    fig.savefig(save_to_map_c)
    plt.close()
    print(f"Saved gain map for the {c_label} channel to '{save_to_map_c}'")

# Plot a histogram
fig, axs = plt.subplots(ncols=len(files), nrows=3, figsize=(5.1, 2), tight_layout=True, gridspec_kw={"wspace":0, "hspace":0}, sharex=True, sharey=True)

# Loop over the cameras
for label, ax_arr, data_RGBG in zip(labels, axs.T, data_RGBG_arrays):
    # Plot the RGB data
    plot.histogram_RGB(data_RGBG, axs=ax_arr, xmin=0.4, xmax=2.8, nrbins=250, xlabel="Gain (ADU/e$^-$)", skip_combined=True)

    # Add a title to the top plot in each column
    ax_arr[0].set_title(label)

# Remove ticks from the left y-axis of all plots except the left-most
for ax in axs[:,1:].ravel():
    ax.tick_params(left=False)

# Add ticks to the right y-axis of the right-most plot
for ax in axs[:,-1].ravel():
    ax.tick_params(right=True, labelright=True)

# Add a label to y-axis of the left-most and right-most, middle plots
axs[1,0].set_ylabel("Frequency")
axs[1,-1].yaxis.set_label_position("right")
axs[1,-1].set_ylabel("Frequency")

# Plot parameters (shared)
axs[0,0].set_yticks([0.5, 1.5])
axs[0,0].set_xticks(np.arange(0.5, 3, 0.5))

# Save the figure
save_to_histogram = save_folder/"gain_histogram.pdf"
plot._saveshow(save_to_histogram)
print(f"Saved RGB histogram to '{save_to_histogram}'")
