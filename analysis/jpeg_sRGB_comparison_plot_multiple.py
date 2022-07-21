"""
Characterise the JPEG response of multiple cameras to multiple sRGB models
(fixed and free gamma).

A plot is generated with a histogram of best-fitting gamma (one) as well as the
relative RMS difference between an sRGB model with a fixed gamma and data (any
number). The number of rows in this plot is equal to the number of cameras.
The number of columns is equal to 1+G, with G the number of different fixed-
gamma models found. If no data is found for a particular combination of
camera and subplot, that subplot is left empty.

Command line arguments:
    * `folder`: folders containing the free-gamma fitting and/or fixed-gamma
    comparison results. These should be NPY files generated using
    jpeg_sRGB_gamma_fixed.py and/or jpeg_sRGB_gamma_fixed.py, respectively.
    It is not necessary for each camera to have data for each (or free) gamma
    value.
    (multiple arguments possible)
"""

import numpy as np
from sys import argv
from spectacle import io, plot
from matplotlib import pyplot as plt

# Get the data folders from the command line
folders = io.path_from_input(argv)
roots = [io.find_root_folder(folder) for folder in folders]
save_to = io.results_folder/"jpeg_sRGB_comparison.pdf"

# Load Camera objects
cameras = [io.load_camera(root) for root in roots]
print(f"Loaded Camera objects: {cameras}")

# Loop through the given folders to see which gamma values are available
gammas_all = []
for folder in folders:
    gamma_files = list(folder.glob("sRGB_comparison_gamma*.npy"))
    gammas = [float(io.split_path(file, "gamma")) for file in gamma_files]
    gammas_all.extend(gammas)

# Convert to a set to get only unique values
gammas_all = set(gammas_all)

# Define bins
bins_gamma = np.linspace(1.75, 2.65, 1000)
bins_RMS = np.linspace(0, 35, 500)

# Plotting functions
def add_RGB_histogram(data, ax, alpha=0.7, density=False, **kwargs):
    for j, c in enumerate(plot.RGB_OkabeIto):
        data_flat = data[...,j].ravel()
        data_flat = data_flat[~np.isnan(data_flat)]
        ax.hist(data_flat, color=c, edgecolor="none", alpha=alpha, density=density, **kwargs)

def load_and_plot_data(path, index, ax, factor=1., **kwargs):
    try:
        data = np.load(path)
    except FileNotFoundError:
        pass
    else:
        data_to_plot = factor * data[index]
        add_RGB_histogram(data_to_plot, ax, **kwargs)

# Create a figure to hold all subplots
number_of_rows = len(cameras)
number_of_columns = 1 + len(gammas_all)
fig, axs = plt.subplots(nrows=number_of_rows, ncols=number_of_columns, sharex="col", sharey="row", tight_layout=True, gridspec_kw={"wspace":0.04, "hspace":0.04*5.1/4.5}, figsize=(5.1, 4.5))

# Loop over the folders and rows of subplots
print("Plotting histogram...")
for ax_row, folder, camera in zip(axs, folders, cameras):

    # Add the camera name to the y-axis of the left-most subplot
    ax_row[0].set_ylabel(camera.name)

    # Generate expected filenames for all combinations of folder and gamma
    gamma_fixed_filenames = [folder/f"sRGB_comparison_gamma{gamma}.npy" for gamma in gammas_all]

    # Plot the best-fitting gamma in the first plot of this row
    try:
        data = np.load(folder/"sRGB_model_free.npy")
    except FileNotFoundError:
        pass
    else:
        gamma = data[1]
        add_RGB_histogram(gamma, ax_row[0], bins=bins_gamma)

    # Loop over the fixed-gamma files and plot these if available
    for ax, file in zip(ax_row[1:], gamma_fixed_filenames):
        try:
            data = np.load(file)
        except FileNotFoundError:
            # If this file does not exist, plot an empty plot
            pass
        else:
            # If the file was loaded, plot a histogram
            RMS_relative = 100 * data[3]  # convert to percentages
            add_RGB_histogram(RMS_relative, ax, bins=bins_RMS)

    print(camera)

# Plot parameters
axs[-1,0].set_xlabel("Best fit $\gamma$")
axs[-1,0].set_xlim(bins_gamma[0], bins_gamma[-1])
axs[-1,0].set_xticks([1.75, 2.00, 2.25, 2.50])
for ax in axs[-1,1:]:  # Set xlabels and xticks on bottom row of fixed-gamma RMS panels
    ax.set_xlabel("RMS diff. [%]")
    ax.set_xlim(bins_RMS[0], bins_RMS[-1])
    ax.set_xticks([0, 10, 20, 30])
for ax, gamma in zip(axs[0,1:], gammas_all):  # Add gamma to titles of top row of fixed-gamma RMS panels
    ax.set_title(r"$\gamma =" + str(gamma) + "$")
for ax in axs[:,0].ravel():  # Fix yticks
    ax.locator_params(axis="y", nbins=4)
for ax in axs[:,1:].ravel():  # Remove y axis labels on all but the left-most panels
    ax.tick_params(axis="y", left=False)
for ax in axs[:-1].ravel():  # Remove x axis labels on all but the bottom panels
    ax.tick_params(axis="x", bottom=False)
for ax in axs.ravel():  # Add a grid everywhere
    ax.grid(True, ls="--", alpha=0.4)

plt.savefig(save_to, bbox_inches="tight")
plt.close()
