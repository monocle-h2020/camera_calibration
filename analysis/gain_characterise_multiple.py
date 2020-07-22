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
from spectacle.general import gauss_nan

# Get the data folder from the command line
files = io.path_from_input(argv)
roots = [io.find_root_folder(file) for file in files]
save_folder = io.results_folder

# Load Camera object
cameras = [io.load_camera(root) for root in roots]
print("Loaded all metadata")

# Find the ISO speed for each gain map, to include in the plot titles
isos = [io.split_iso(file) for file in files]

# Load the data
data_arrays = [np.load(file) for file in files]
print("Loaded data")

# Demosaick the data (split into the Bayer RGBG2 channels)
data_RGBG_arrays = [camera.demosaick(data) for data, camera in zip(data_arrays, cameras)]
print("Demosaicked data")

# Convolve the RGBG2 data with a Gaussian kernel
data_RGBG_gauss_arrays = [gauss_nan(RGBG, sigma=(0,5,5)) for RGBG in data_RGBG_arrays]
print("Gaussed data")

# Plot a Gaussed map for each channel
# Loop over the Bayer RGBG2 channels
for j, c in enumerate(plot.RGBG2):
    # Create a figure to plot into
    fig, axs = plt.subplots(ncols=len(files), figsize=(3*len(files), 2.3), squeeze=True, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0})
    # Loop over the demosaicked/gaussed gain maps and plot them into the figure
    for camera, iso, ax, data_RGBG, data_RGBG_gauss in zip(cameras, isos, axs, data_RGBG_arrays, data_RGBG_gauss_arrays):

        # Plot channel j into this figure
        im = ax.imshow(data_RGBG_gauss[j], cmap=plot.cmaps[c+"r"])

        # Plot parameters
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{camera.device.name} (ISO {iso})")

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
        print(f"{camera.device.name:<10}: ISO {iso:>4}")
        print(f"{c:>2}: {percentile_low:.2f} -- {percentile_high:.2f}")

    # Save the figure
    save_to_map_c = save_folder/f"gain_map_{c}.pdf"
    fig.savefig(save_to_map_c)
    plt.close()
    print(f"Saved gain map for the {c} channel to '{save_to_map_c}'")

# Plot a histogram
bins = np.linspace(0.4, 2.8, 250)
fig, axs = plt.subplots(ncols=len(files), nrows=3, figsize=(3*len(files), 2.3), tight_layout=True, gridspec_kw={"wspace":0, "hspace":0}, sharex=True, sharey=True)

# Loop over the cameras
for camera, iso, ax_arr, data_RGBG in zip(cameras, isos, axs.T, data_RGBG_arrays):

    # Combine the G and G2 channels and remove NaN values
    R = data_RGBG[0].ravel()    ; R = R[~np.isnan(R)]
    G = data_RGBG[1::2].ravel() ; G = G[~np.isnan(G)]
    B = data_RGBG[2].ravel()    ; B = B[~np.isnan(B)]

    # Plot the RGB data
    for ax, D, c in zip(ax_arr, [R, G, B], plot.RGB):
        ax.hist(D, bins=bins, color=c, edgecolor=c, density=True)
        ax.grid(True)

        # Remove ticks from the left y-axis of all plots except the left-most
        if ax not in axs[:,0]:
            ax.tick_params(left=False)

        # Add ticks to the right y-axis of the right-most plot
        if ax in axs[:,-1]:
            ax.tick_params(right=True, labelright=True)

    # Add a title to the top plot in each column
    ax_arr[0].set_title(f"{camera.device.name} (ISO {iso})")

    # Add a label to the x-axis of the bottom plot in each column
    ax_arr[-1].set_xlabel("Gain (ADU/e$^-$)")

# Add a label to y-axis of the left-most and right-most, middle plots
axs[1,0].set_ylabel("Frequency")
axs[1,-1].yaxis.set_label_position("right")
axs[1,-1].set_ylabel("Frequency")

# Plot parameters (shared)
axs[0,0].set_xlim(bins[0], bins[-1])
axs[0,0].set_yticks([0.5, 1.5])
axs[0,0].set_xticks(np.arange(0.5, 3, 0.5))

# Save the figure
save_to_histogram = save_folder/"gain_histogram.pdf"
fig.savefig(save_to_histogram)
plt.close()
print(f"Saved RGB histogram to '{save_to_histogram}'")
