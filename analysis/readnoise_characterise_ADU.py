"""
Analyse read noise maps (in ADU) generated using the calibration functions.

Command line arguments:
    * `folder`: folder containing NPY stacks of bias data taken at different
    ISO speeds.
"""

from sys import argv
from spectacle import io, analyse

# Get the data folder from the command line
folder = io.path_from_input(argv)
root = io.find_root_folder(folder)
save_to = root/"analysis/readnoise"

# Get metadata
camera = io.load_camera(root)

# Load the data
isos, stds = io.load_stds(folder, retrieve_value=io.split_iso)

# Print statistics at each ISO
stats = analyse.statistics(stds, prefix_column=isos, prefix_column_header="ISO")
print(stats)

# Range on the x axis for the histograms
xmin, xmax = 0., analyse.symmetric_percentiles(stds)[1]

# Loop over the data and make plots at each ISO value
for ISO, std in zip(isos, stds):
    save_to_histogram = save_to/f"readnoise_ADU_histogram_iso{ISO}.pdf"
    save_to_maps = save_to/f"readnoise_ADU_map_iso{ISO}.pdf"

    camera.plot_histogram_RGB(std, xmin=xmin, xmax=xmax, xlabel="Read noise (ADU)", saveto=save_to_histogram)
    camera.plot_gauss_maps(std, colorbar_label="Read noise (ADU)", saveto=save_to_maps)

    print(f"Saved plots for ISO speed {ISO}")
