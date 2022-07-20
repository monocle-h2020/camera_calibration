"""
Compare two flat-field correction maps, from data or modelled.

Command line arguments:
    * `file1`: the location of the first flat-field map.
    * `file2`: the location of the second flat-field map.
    These flat-field maps should be NPY stacks representing data or generated using ../calibration/flatfield.py

To do:
    * Input labels for plots
    * Individual DNG files as input
    * Statistics as function of radius from optical centre
    * Different statistics (e.g. MAD instead of RMS)
"""
from sys import argv
from matplotlib import pyplot as plt
import numpy as np
from spectacle import io, analyse, flat, plot
from spectacle.general import RMS

# Get the data folder from the command line
file1, file2 = io.path_from_input(argv)
root = io.find_root_folder(file1)
label = "comparison_" + file1.stem + "_X_" + file2.stem

# Load Camera object
camera = io.load_camera(root)
print(f"Loaded Camera object: {camera}")

# Save locations
savefolder = camera.filename_analysis("flatfield", makefolders=True)
save_to_histogram = savefolder/f"{label}_histogram.pdf"
save_to_histogram_RGB = savefolder/f"{label}_histogram_RGB.pdf"
save_to_maps = savefolder/f"{label}_map.pdf"
save_to_combined_map = savefolder/f"{label}_map_combined.pdf"

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
plt.figure(figsize=(4,2), tight_layout=True)
plt.hist(difference.ravel(), bins=250, color='k')
plt.xlabel("Difference in correction factor $\Delta g$")
plt.ylabel("Counts")
plt.grid(True, ls="--")
plt.savefig(save_to_histogram)
plt.close()
print(f"Saved histogram to '{save_to_histogram}'")

# Plot an RGB histogram of the difference between the maps
camera.plot_histogram_RGB(difference, xlabel="Difference in correction factor $\Delta g$", saveto=save_to_histogram_RGB)
print(f"Saved RGB histogram to '{save_to_histogram_RGB}'")

# Make Gaussian maps of the difference between data and model
camera.plot_gauss_maps(difference, colorbar_label="$\Delta g$", saveto=save_to_maps)
print(f"Saved Gaussian maps to '{save_to_maps}'")

# Plot both maps and the difference between them
diff_max = np.nanmax(np.abs(flat.clip_data(difference)))
shared_max = max([np.nanmax(flat.clip_data(m)) for m in (map1, map2)])
vmins = [1, 1, -diff_max]
vmaxs = [shared_max, shared_max, diff_max]
clabels = ["$g$ (Observed)", "$g$ (Best fit)", "Difference"]
fig, axs = plt.subplots(ncols=3, figsize=(6.5,2), sharex=True, sharey=True, squeeze=True, tight_layout=True, gridspec_kw={"wspace":0.02, "hspace":0})
for data, ax, vmin, vmax, clabel in zip([map1, map2, difference], axs, vmins, vmaxs, clabels):
    img = ax.imshow(data, vmin=vmin, vmax=vmax, cmap="cividis")
    ax.set_xticks([])
    ax.set_yticks([])
    colorbar_here = plot.colorbar(img)
    colorbar_here.set_label(clabel)
    colorbar_here.locator = plot.ticker.MaxNLocator(nbins=4)
    colorbar_here.update_ticks()
plt.savefig(save_to_combined_map, bbox_inches="tight", dpi=400)
plt.close()
print(f"Saved combined map to '{save_to_combined_map}'")
