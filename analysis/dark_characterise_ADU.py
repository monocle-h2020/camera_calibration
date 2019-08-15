"""
Analyse dark current maps (in ADU/s) generated using the calibration functions.

Command line arguments:
    * `file`: the location of the dark current map to be analysed.
"""

import numpy as np
from sys import argv
from spectacle import io, analyse

# Get the data file from the command line
file = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(file)

# Get metadata
phone = io.load_metadata(root)
colours = io.load_colour(stacks)

# Load the data
dark_current = np.load(file)
print("Loaded data")

# Convolve the map with a Gaussian kernel and plot an image of the result
analyse.plot_gauss_maps(dark_current, colours, colorbar_label="Dark current (norm. ADU/s)", saveto=root/f"results/dark/dark_current_map.pdf")
print("Saved Gauss map")

# Range on the x axis for the histogram
xmin, xmax = analyse.symmetric_percentiles(dark_current, percent=0.001)

# Split the data into the RGBG2 filters and make histograms (aggregate and per
# filter)
analyse.plot_histogram_RGB(dark_current, colours, xlim=(xmin, xmax), xlabel="Dark current (norm. ADU/s)", saveto=root/f"results/dark/dark_current_histogram.pdf")
print("Saved RGB histogram")

# Check how many pixels are over some threshold in dark current
threshold = 50
pixels_over_threshold = np.where(np.abs(dark_current) > threshold)
number_over_threshold = len(pixels_over_threshold[0])
print(f"There are {number_over_threshold} pixels with a dark current >{threshold} normalised ADU/s.")
