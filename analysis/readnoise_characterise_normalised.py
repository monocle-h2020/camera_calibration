"""
Analyse read noise maps (in normalised ADU) generated using the calibration
functions.

This script requires an ISO normalisation look-up table to have been generated.

Command line arguments:
    * `folder`: folder containing NPY stacks of bias data taken at different
    ISO speeds.
"""

from sys import argv
from spectacle import io, analyse, calibrate

# Get the data folder from the command line
folder = io.path_from_input(argv)
root = io.find_root_folder(folder)
save_to = root/"analysis/readnoise"

# Get metadata
camera = io.load_metadata(root)

# Load the data
isos, stds = io.load_stds(folder, retrieve_value=io.split_iso)

# Normalise the data using the ISO look-up table
stds_normalised = calibrate.normalise_iso(root, isos, stds)

# Print statistics at each ISO
stats = analyse.statistics(stds_normalised, prefix_column=isos, prefix_column_header="ISO")
print(stats)

# Range on the x axis for the histograms
xmin, xmax = 0., analyse.symmetric_percentiles(stds_normalised)[1]

# Loop over the data and make plots at each ISO value
for ISO, std in zip(isos, stds_normalised):
    save_to_histogram = save_to/f"readnoise_normalised_histogram_iso{ISO}.pdf"
    save_to_maps = save_to/f"readnoise_normalised_map_iso{ISO}.pdf"

    camera.plot_histogram_RGB(std, xmin=xmin, xmax=xmax, xlabel="Read noise (norm. ADU)", saveto=save_to_histogram)
    camera.plot_gauss_maps(std, colorbar_label="Read noise (norm. ADU)", saveto=save_to_maps)

    print(f"Saved plots for ISO speed {ISO}")
