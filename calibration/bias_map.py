"""
Create a bias map using the mean bias (zero-light, shortest-exposure images)
images. Bias data for all ISOs are loaded, but the map is only saved for the
lowest ISO.

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

# Load the mean stacks for each ISO value
isos, means = io.load_means(folder, retrieve_value=io.split_iso)

# Find the lowest ISO value in the data and select the respective bias map
lowest_iso = isos.argmin()
bias_map = means[lowest_iso]

# Save the bias map to the `products` folder
np.save(products/"bias.npy", bias_map)
