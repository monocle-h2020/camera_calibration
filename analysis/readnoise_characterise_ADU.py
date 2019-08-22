"""
Analyse read noise maps (in ADU) generated using the calibration functions.

Command line arguments:
    * `folder`: the folder containing the read noise maps to be analysed.
"""

from sys import argv
from spectacle import io, analyse

# Get the data folder from the command line
folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
save_to = root/"results/readnoise"

# Get metadata
camera = io.load_metadata(root)

# Load the data
isos, stds = io.load_stds(folder, retrieve_value=io.split_iso)

# Print statistics at each ISO
stats = analyse.statistics(stds, prefix_column=isos, prefix_column_header="ISO")
print(stats)

# Range on the x axis for the histograms
xmin, xmax = 0, analyse.symmetric_percentiles(stds, percent=0.001)[1]

# Loop over the data and make plots at each ISO value
for ISO, std in zip(isos, stds):
    save_to_histogram = save_to/f"readnoise_ADU_histogram_iso{ISO}.pdf"
    save_to_maps = save_to/f"readnoise_ADU_map_iso{ISO}.pdf"

    analyse.plot_histogram_RGB(std, camera.bayer_map, xlim=(xmin, xmax), xlabel="Read noise (norm. ADU)", saveto=save_to_histogram)
    analyse.plot_gauss_maps(std, camera.bayer_map, colorbar_label="Read noise (norm. ADU)", saveto=save_to_maps)

    print(f"Saved plots for ISO speed {ISO}")
