"""
Analyse dark current maps (in normalised ADU/s) generated using the calibration
functions.

Command line arguments:
    * `file`: the location of the dark current map to be analysed.
"""

import numpy as np
from sys import argv
from spectacle import io, analyse

# Get the data file from the command line
file = io.path_from_input(argv)
root = io.find_root_folder(file)
save_folder = root/f"results/dark/"

# Get metadata
camera = io.load_metadata(root)

# Load the data
dark_current = np.load(file)
print("Loaded data")

# Convolve the map with a Gaussian kernel and plot an image of the result
save_to_maps = save_folder/"dark_current_map_ADU.pdf"
analyse.plot_gauss_maps(dark_current, camera.bayer_map, colorbar_label="Dark current (norm. ADU/s)", saveto=save_to_maps)
print(f"Saved Gauss map to '{save_to_maps}'")

# Range on the x axis for the histogram
xmin, xmax = analyse.symmetric_percentiles(dark_current, percent=0.001)

# Split the data into the RGBG2 filters and make histograms (aggregate and per
# filter)
save_to_histogram = save_folder/"dark_current_histogram_ADU.pdf"
analyse.plot_histogram_RGB(dark_current, camera.bayer_map, xlim=(xmin, xmax), xlabel="Dark current (norm. ADU/s)", saveto=save_to_histogram)
print(f"Saved RGB histogram to '{save_to_histogram}'")

# Check how many pixels are over some threshold in dark current
threshold = 50
pixels_over_threshold = np.where(np.abs(dark_current) > threshold)
number_over_threshold = len(pixels_over_threshold[0])
print(f"There are {number_over_threshold} pixels with a dark current >{threshold} normalised ADU/s.")
