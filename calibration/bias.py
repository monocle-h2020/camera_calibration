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
root = io.find_root_folder(folder)

# Load the mean stacks for each ISO value
isos, means = io.load_means(folder, retrieve_value=io.split_iso)
print(f"Loaded bias data for {len(isos)} ISO values from '{folder}'")

# Find the lowest ISO value in the data and select the respective bias map
lowest_iso_index = isos.argmin()
bias_map = means[lowest_iso_index]

# Save the bias map to the `products` folder
save_to = root/"products/bias.npy"
np.save(save_to, bias_map)
print(f"Saved bias map at ISO {isos[lowest_iso_index]} to '{save_to}'")
