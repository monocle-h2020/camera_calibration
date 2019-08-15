"""
Analyse bias maps (in ADU) generated using the calibration functions.

Command line arguments:
    * `folder`: the folder containing the bias maps to be analysed.
"""

from sys import argv
from spectacle import io, analyse

# Get the data folder from the command line
folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)

# Get metadata
phone = io.load_metadata(root)
colours = io.load_colour(stacks)
savefolder = results/"bias"

# Load the data
isos, means = io.load_means(folder, retrieve_value=io.split_iso)
print("Loaded data")

# Print statistics of the bias at each ISO
stats = analyse.statistics(means, prefix_column=isos, prefix_column_header="ISO")
print(stats)

# Range on the x axis for the histograms
xmax = phone["software"]["bias"] + 25
xmin = max(phone["software"]["bias"] - 25, 0)

# Loop over the data and make plots at each ISO value
for ISO, mean in zip(isos, means):
    saveto_histogram = savefolder/f"bias_histogram_iso{ISO}.pdf"
    saveto_maps = savefolder/f"bias_map_iso{ISO}.pdf"
    
    analyse.plot_histogram_RGB(mean, colours, xlim=(xmin, xmax), xlabel="Bias (ADU)", saveto=saveto_histogram)
    analyse.plot_gauss_maps(mean, colours, colorbar_label="Bias (ADU)", saveto=saveto_maps)
