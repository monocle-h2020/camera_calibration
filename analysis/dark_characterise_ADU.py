"""
Analyse dark current maps (in ADU/s) generated using the calibration functions.

Command line arguments:
    * `file`: the location of the dark current map to be analysed.
"""

import numpy as np
from sys import argv
from spectacle import raw, plot, io, analyse
from spectacle.general import gaussMd

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
dark_current_gauss = gaussMd(dark_current, sigma=10)
plot.show_image(dark_current_gauss, colorbar_label="Dark current (norm. ADU/s)", saveto=root/f"results/dark/map.pdf")
print("Saved Gauss map")

# Split the data into the RGBG2 filters and make histograms (aggregate and per
# filter)
dark_RGBG, _= raw.pull_apart(dark_current, colours)
plot.histogram_RGB(dark_RGBG, xlim=(-25, 50), xlabel="Dark current (norm. ADU/s)", saveto=root/f"results/dark/histogram_RGB.pdf")
print("Saved RGB histogram")

# Convolve the data in each filter (RGBG2) with a Gaussian kernel and plot
# images of the result
dark_RGBG_gauss = gaussMd(dark_RGBG, sigma=(0,5,5))
vmin, vmax = dark_RGBG_gauss.min(), dark_RGBG_gauss.max()
plot.show_RGBG(dark_RGBG_gauss, colorbar_label="Dark current (norm. ADU/s)", saveto=root/f"results/dark/map_RGBG2.pdf", vmin=vmin, vmax=vmax)
for j, c in enumerate("RGBG"):
    X = "2" if j == 3 else ""
    plot.show_image(dark_RGBG_gauss[j], colorbar_label="Dark current (norm. ADU/s)", saveto=root/f"results/dark/map_{c}{X}.pdf", colour=c, vmin=vmin, vmax=vmax)

# Print statistics of the dark current per filter
RGBG2 = ["R", "G", "B", "G2"]
stats = analyse.statistics(dark_RGBG, prefix_column=RGBG2, prefix_column_header="Filter")
print(stats)

# Check how many pixels are over some threshold in dark current
threshold = 50
pixels_over_threshold = np.where(np.abs(dark_current) > threshold)
number_over_threshold = len(pixels_over_threshold[0])
print(f"There are {number_over_threshold} pixels with a dark current >{threshold} normalised ADU/s.")
