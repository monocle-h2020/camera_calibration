"""
Determines the linearity of the response of each pixel in a camera using JPEG
data. The Pearson r coefficient is used as a measure of linearity. Two methods
for measuring linearity are currently supported:
    * "polarisers" (`p` on the command-line), two linear polarisers with
    varying angles.
    * "exposure_times" (`t` on the command-line), the same image taken with
    varying exposure times.

Command line arguments:
    * `folder`: folder containing stacked linearity data
    * `mode`: the calibration mode (`p` for polarisers, `t` for exposure times)

To do:
    * Implement a more general interface for data obtained using different
    methods.
"""

import numpy as np
from sys import argv
from spectacle import io, linearity as lin

# Hacky solution to get the mode from the command-line
# To-do: replace with optparse
mode = argv[2]
argv = argv[:2]
calibration_mode = lin.calibration_mode(mode)
print("Selected linearity calibration mode:", calibration_mode)

# Get the data folder from the command line
folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)

phone = io.load_metadata(root)
max_value = 2**phone["camera"]["bits"]
saturation = 0.95 * max_value

if calibration_mode == "polarisers":
    # Load the mean stacks for each polariser angle
    angles, jmeans = io.load_jmeans(folder, retrieve_value=io.split_pol_angle)
    print("Read data")

    # Convert polariser angles to intensities using the observed angle between
    # polarisers
    offset_angle = io.load_angle(stacks)
    intensities = lin.malus(angles, offset_angle)

    # Errors are currently not used
    intensities_errors = lin.malus_error(angles, offset_angle, sigma_angle0=1, sigma_angle1=1)

elif calibration_mode == "exposure_time":
    # Load the mean stacks for each exposure time
    intensities, jmeans = io.load_jmeans(folder, retrieve_value=io.split_exposure_time)
    print("Read data")

    # Errors are currently not used
    intensities_errors = np.zeros_like(intensities)

# Calculate the Pearson r value for each pixel
print("Calculating Pearson r...", end=" ", flush=True)
r, saturated = lin.calculate_pearson_r_values_jpeg(intensities, jmeans, saturate=saturation)
print("... Done!")

# Save the results
save_to = root/"products/linearity_jpeg.npy"
np.save(save_to, r)
print(f"Saved results to '{save_to}'")
