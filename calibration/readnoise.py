"""
Create a read noise map using the standard deviation bias (zero-light, shortest-exposure images) images.
Data for all ISOs are loaded, but the map is only saved for the lowest ISO.

Command line arguments:
    * `folder`: folder containing NPY stacks of bias data taken at different ISO speeds.

Example:
    python calibration/readnoise.py ~/SPECTACLE_data/iPhone_SE/stacks/bias_readnoise/

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
save_to = camera.filename_calibration("readnoise.npy")

# Load the standard deviation stacks for each ISO value
isos, stds = io.load_stds(folder, retrieve_value=io.split_iso)
print(f"Loaded bias data for {len(isos)} ISO values from '{folder}'")

# Find the lowest ISO value in the data and select the respective read noise map
lowest_iso_index = isos.argmin()
readnoise_map = stds[lowest_iso_index]

# Save the read noise map
np.save(save_to, readnoise_map)
print(f"Saved read noise map at ISO {isos[lowest_iso_index]} to '{save_to}'")
