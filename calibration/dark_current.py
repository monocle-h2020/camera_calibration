"""
Create a dark current map using dark data (zero light, varying exposure times).
A map in ADU/s is created.

Command line arguments:
    * `folder`: folder containing stacked dark data.

To do:
    * Save maps for all ISOs and use these in the calibration process.
"""

import numpy as np
from sys import argv
from spectacle import io, calibrate, dark

# Get the data folder from the command line
folder = io.path_from_input(argv)
root = io.find_root_folder(folder)

# Get the ISO speed at which the data were taken from the folder name
ISO = io.split_iso(folder)

# Load the data
times, means = io.load_means(folder, retrieve_value=io.split_exposure_time)
print(f"Loaded data at {len(times)} exposure times")

# Fit a linear trend to each pixel
dark_current, bias_fit = dark.fit_dark_current_linear(times, means)
print("Fitted dark current to each pixel")

save_to = root/"products/dark_current.npy"
np.save(save_to, dark_current)
print(f"Saved dark current map at ISO {ISO} to '{save_to}'")

# ISO normalisation
dark_current_normalised = calibrate.normalise_iso(root, dark_current, ISO)
save_to_normalised = root/"products/dark_current_normalised.npy"
np.save(save_to_normalised, dark_current_normalised)
print(f"Saved normalised dark current map to '{save_to_normalised}'")
