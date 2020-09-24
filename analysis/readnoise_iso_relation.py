"""
Plot the relationship between ISO speed and read noise, based on read noise
maps generated at various ISO speeds.

This script requires an ISO normalisation look-up table to have been generated.

Command line arguments:
    * `folder`: folder containing NPY stacks of bias data taken at different
    ISO speeds.
"""

from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, analyse, iso

# Get the data folder from the command line
folder = io.path_from_input(argv)
root = io.find_root_folder(folder)

# Load Camera object
camera = io.load_camera(root)
print(f"Loaded Camera object: {camera}")

# Save locations
savefolder = camera.filename_analysis("readnoise", makefolders=True)
save_to_plot = savefolder/"readnoise_ISO_relation.pdf"

# Load the data
isos, stds = io.load_stds(folder, retrieve_value=io.split_iso)

# Normalise the data using the ISO look-up table
stds_normalised = camera.normalise_iso(isos, stds)

# Print statistics at each ISO
stats = analyse.statistics(stds_normalised, prefix_column=isos, prefix_column_header="ISO")
print(stats)

# Plot the mean read noise as a function of ISO speed
std_mean = stds_normalised.mean(axis=(1,2))
std_std = stds_normalised.std(axis=(1,2))

xmax = iso.get_max_iso(camera)
plt.figure(figsize=(3,2), tight_layout=True)
plt.errorbar(isos, std_mean, yerr=std_std, fmt="ko")
plt.xlabel("ISO speed")
plt.ylabel("Mean read noise\n(norm. ADU)")
plt.xlim(0, xmax)
plt.ylim(ymin=0)
plt.grid(True, ls="--", alpha=0.3)
plt.savefig(save_to_plot)
plt.close()
print(f"Saved plot to '{save_to_plot}'")
