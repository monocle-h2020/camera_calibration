"""
Analyse dark current maps (in normalised ADU/s) generated using the calibration
functions. The dark current is converted from normalised ADU/s to electrons/s
using a gain map.

Command line arguments:
    * `file`: the location of the dark current map to be analysed.
"""


import numpy as np
from sys import argv
from spectacle import io, analyse, calibrate

# Get the data file from the command line
file = io.path_from_input(argv)
root = io.find_root_folder(file)
save_folder = root/f"analysis/dark_current/"

# Get metadata
camera = io.load_metadata(root)

# Load the data
dark_current_normADU = np.load(file)
print("Loaded data")

# Convert the data to photoelectrons per second
dark_current_electrons = calibrate.convert_to_photoelectrons(root, dark_current_normADU)

# Convolve the map with a Gaussian kernel and plot an image of the result
save_to_maps = save_folder/"dark_current_map_electrons.pdf"
analyse.plot_gauss_maps(dark_current_electrons, camera.bayer_map, colorbar_label="Dark current (e-/s)", saveto=save_to_maps)
print(f"Saved Gauss map to '{save_to_maps}'")

# Range on the x axis for the histogram
xmin, xmax = analyse.symmetric_percentiles(dark_current_electrons, percent=0.001)

# Split the data into the RGBG2 filters and make histograms (aggregate and per
# filter)
save_to_histogram = save_folder/"dark_current_histogram_ADU.pdf"
analyse.plot_histogram_RGB(dark_current_electrons, camera.bayer_map, xlim=(xmin, xmax), xlabel="Dark current (e-/s)", saveto=save_to_histogram)
print(f"Saved RGB histogram to '{save_to_histogram}'")
