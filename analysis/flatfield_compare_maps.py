"""
Compare two flat-field correction maps, from data or modelled.

Command line arguments:
    * `file1`: the location of the first flat-field map.
    * `file2`: the location of the second flat-field map.
    These flat-field maps should be NPY stacks generated using
    ../calibration/flatfield.py

To do:
    * Input labels for plots
"""

import numpy as np
from sys import argv
from spectacle import io, analyse, plot
from spectacle.general import RMS
from matplotlib import pyplot as plt

# Get the data folder from the command line
file1, file2 = io.path_from_input(argv)
root = io.find_root_folder(file1)
savefolder = root/"analysis/flatfield/"
label = "comparison_" + file1.stem + "_X_" + file2.stem

# Get metadata
camera = io.load_camera(root)
print("Loaded metadata")

# Load the data
map1 = np.load(file1)
map2 = np.load(file2)
print("Loaded data")

# Check that the two maps have the same shape
assert map1.shape == map2.shape, f"Mis-match in shape of maps: {map1.shape} vs. {map2.shape}"

# Calculate the difference between the maps
difference = map1 - map2
print("Calculated difference map")

# Print some statistics
print(f"RMS difference: {RMS(difference):.3f}")

# Plot a histogram of the difference between the maps
save_to_histogram = savefolder/f"{label}_histogram.pdf"
plt.figure(figsize=(4,2), tight_layout=True)
plt.hist(difference.ravel(), bins=250, color='k')
plt.xlabel("Difference in correction factor $\Delta g$")
plt.ylabel("Counts")
plt.grid(True, ls="--")
plt.savefig(save_to_histogram)
plt.close()
print(f"Saved histogram to '{save_to_histogram}'")

# Plot an RGB histogram of the difference between the maps
save_to_histogram_RGB = savefolder/f"{label}_histogram_RGB.pdf"
camera.plot_histogram_RGB(difference, xlabel="Difference in correction factor $\Delta g$", saveto=save_to_histogram_RGB)
print(f"Saved RGB histogram to '{save_to_histogram_RGB}'")

# Make Gaussian maps of the difference between data and model
save_to_maps = savefolder/f"{label}_map.pdf"
camera.plot_gauss_maps(difference, colorbar_label="$\Delta g$", saveto=save_to_maps)
print(f"Saved Gaussian maps to '{save_to_maps}'")

# Plot both maps and the difference between them
save_to_combined_map = savefolder/f"{label}_map_combined.pdf"
diff_max = max([abs(difference.max()), abs(difference.min())])
shared_max = max([map1.max(), map2.max()])
vmins = [1, 1, -diff_max]
vmaxs = [shared_max, shared_max, diff_max]
clabels = ["$g$ (map 1)", "$g$ (map 2)", "Difference"]
fig, axs = plt.subplots(ncols=3, figsize=(6,2), sharex=True, sharey=True, squeeze=True, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0})
for data, ax, vmin, vmax, clabel in zip([map1, map2, difference], axs, vmins, vmaxs, clabels):
    img = ax.imshow(data, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    colorbar_here = plot.colorbar(img)
    colorbar_here.set_label(clabel)
    colorbar_here.locator = plot.ticker.MaxNLocator(nbins=4)
    colorbar_here.update_ticks()
plt.savefig(save_to_combined_map)
plt.close()
print(f"Saved combined map to '{save_to_combined_map}'")
