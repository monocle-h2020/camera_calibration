"""
Characterise an sRGB model with fixed gamma fitted to JPEG linearity data using
the `jpeg_sRGB_gamma_fixed.py` script.

Command line arguments:
    * `file`: the file containing the comparison results.
"""

import numpy as np
from sys import argv
from spectacle import io, plot, analyse
from matplotlib import pyplot as plt

# Get the data folder from the command line
file = io.path_from_input(argv)
root = io.find_root_folder(file)
savefolder = root/"analysis/jpeg/"

# Load the data
normalisations, Rsquares, RMSes, RMSes_relative = np.load(file)
RMSes_percent = RMSes_relative * 100.
print("Loaded data")

# Find the associated gamma value from the filename
gamma = float(io.split_path(file, "gamma"))
print(f"Gamma = {gamma}")

# Labels for the plot
labels = ["Normalisation", "$R^2$", "RMS diff. (DN)", "RMS diff. (%)"]

# Find the number of non-NaN pixels which will be used in the plot
number_of_pixels = normalisations[...,0].size
pixels_not_nan = np.where(~np.isnan(normalisations[...,0]))[0]
number_of_pixels_not_nan = pixels_not_nan.size
percentage_not_nan = 100. * number_of_pixels_not_nan / number_of_pixels
print(f"{number_of_pixels_not_nan}/{number_of_pixels} ({percentage_not_nan:.1f}%) of pixels are not NaN.")

# Create a figure to hold the plot
fig, axs = plt.subplots(ncols=4, figsize=(9,2), sharey=True, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0})

# Loop over the results
for ax, result, label in zip(axs, [normalisations, Rsquares, RMSes, RMSes_percent], labels):
    # Get bins from the range of data
    xmin, xmax = analyse.symmetric_percentiles(result)
    bins = np.linspace(xmin, xmax, 150)

    # Loop over the RGB channels
    for j, c in enumerate(plot.rgb):
        # Flatten the array and remove NaN elements
        result_c = result[...,j].ravel()
        result_c = result_c[~np.isnan(result_c)]
        ax.hist(result_c, bins, color=c, alpha=0.7)

    # Plot parameters
    ax.set_xlabel(label)
    ax.set_xlim(xmin, xmax)

# Plot parameters
axs[0].set_ylabel("Counts")

# Save the figure to file
save_to = savefolder/f"sRGB_comparison_gamma{gamma}.pdf"
plt.savefig(save_to)
plt.close()
print(f"Saved figure to '{save_to}'")
