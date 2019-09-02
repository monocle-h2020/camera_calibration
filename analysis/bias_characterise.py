"""
Analyse bias maps (in ADU) generated using the calibration functions.

Command line arguments:
    * `folder`: folder containing NPY stacks of bias data taken at different
    ISO speeds.
"""

from sys import argv
from spectacle import io, analyse

# Get the data folder from the command line
folder = io.path_from_input(argv)
root = io.find_root_folder(folder)
save_to = root/"analysis/bias/"

# Get metadata
camera = io.load_metadata(root)

# Load the data
isos, means = io.load_means(folder, retrieve_value=io.split_iso)
print("Loaded data")

# Print statistics at each ISO
stats = analyse.statistics(means, prefix_column=isos, prefix_column_header="ISO")
print(stats)

# Range on the x axis for the histograms
xmin, xmax = analyse.symmetric_percentiles(means, percent=0.001)

# Loop over the data and make plots at each ISO value
for ISO, mean in zip(isos, means):
    save_to_histogram = save_to/f"bias_histogram_iso{ISO}.pdf"
    save_to_maps = save_to/f"bias_map_iso{ISO}.pdf"

    analyse.plot_histogram_RGB(mean, camera.bayer_map, xlim=(xmin, xmax), xlabel="Bias (ADU)", saveto=save_to_histogram)
    analyse.plot_gauss_maps(mean, camera.bayer_map, colorbar_label="Bias (ADU)", saveto=save_to_maps)

    print(f"Saved plots for ISO speed {ISO}")
