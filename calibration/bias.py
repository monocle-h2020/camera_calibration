"""
Create a bias map using the mean bias (zero-light, shortest-exposure images) images.
Bias data for all ISOs are loaded, but the map is only saved for the lowest ISO.

Command line arguments:
    * `folder`: folder containing NPY stacks of bias data taken at different ISO speeds.

Example:
    python calibration/bias.py ~/SPECTACLE_data/iPhone_SE/stacks/bias_readnoise/

To do:
    * Save maps for all ISOs and use these in the calibration process.
"""
from sys import argv
import numpy as np
from spectacle import io

# Get the data folder from the command line
folder = io.path_from_input(argv)
root = io.find_root_folder(folder)

# Load Camera object
camera = io.load_camera(root)
print(f"Loaded Camera object: {camera}")

# Save location based on camera name
save_to = camera.filename_calibration("bias.npy")

# Load the mean stacks for each ISO value
isos, means = io.load_means(folder, retrieve_value=io.split_iso)
print(f"Loaded bias data for {len(isos)} ISO values from '{folder}'")

# Find the lowest ISO value in the data and select the respective bias map
lowest_iso_index = isos.argmin()
bias_map = means[lowest_iso_index]

# Save the bias map for calibration purposes
np.save(save_to, bias_map)
print(f"Saved bias map at ISO {isos[lowest_iso_index]} to '{save_to}'")
