"""
Create a read noise map using the standard deviation bias (zero-light,
shortest-exposure images) images. Bias data for all ISOs are loaded, but the
map is only saved for the lowest ISO.

Command line arguments:
    * `folder`: folder containing stacked bias data

To do:
    * Save maps for all ISOs and use these in the calibration process.
"""

import numpy as np
from sys import argv
from spectacle import io

# Get the data folder from the command line
folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)

# Load the standard deviation stacks for each ISO value
isos, stds = io.load_stds(folder, retrieve_value=io.split_iso)

# Find the lowest ISO value in the data and select the respective read noise
# map
lowest_iso = isos.argmin()
readnoise_map = stds[lowest_iso]

# Save the read noise map to the `products` folder
np.save(products/"readnoise.npy", readnoise_map)
