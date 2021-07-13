"""
Create a gain map using gain images, fitting the mean and standard deviation
against each other. Data for a single ISO speed are loaded and fitted.

An ISO speed normalisation is applied to the data. This means this script
requires an ISO speed look-up table to exist.

Command line arguments:
    * `folder`: folder containing NPY stacks of gain data taken at different
    exposure conditions, all with the same ISO speed.
"""

import numpy as np
from sys import argv
from spectacle import io, symmetric_percentiles

# Get the data folder from the command line
folder = io.path_from_input(argv)
root = io.find_root_folder(folder)

# Load Camera object
camera = io.load_camera(root)
print(f"Loaded Camera object: {camera}")

# Save location based on camera name
save_to_normalised_map = camera.filename_calibration("gain.npy")

# Get the ISO speed of these data from the folder name
ISO = io.split_iso(folder)
save_to_original_map = camera.filename_intermediaries(f"gain/gain_map_iso{ISO}.npy", makefolders=True)

# Load the data
names, means = io.load_means(folder)
names, stds = io.load_stds(folder)
print("Loaded data")

# Find pixels near saturation to mask later
fit_max = 0.95 * camera.saturation
mask = (means >= fit_max)

# Bias correction
means = camera.correct_bias(means)

# Use variance instead of standard deviation
variance = stds**2

# Make masked arrays that don't include non-linear values (near saturation)
means = np.ma.array(means, mask=mask)
variance = np.ma.array(variance, mask=mask)
weights = 1/means

# Perform weighted-least-squares
# https://stats.stackexchange.com/a/489949
mean_mean = np.ma.average(means, weights=weights, axis=0)
variance_mean = np.ma.average(variance, weights=weights, axis=0)

gain_map = np.ma.sum(weights * (means - mean_mean) * (variance - variance_mean), axis=0) / np.ma.sum(weights * (means - mean_mean)**2, axis=0)
readnoise_map = variance_mean - gain_map * mean_mean
print("Finished fitting gain map")

# Mask off
gain_map = gain_map.data
readnoise_map = readnoise_map.data

# Filter out bad pixels (outside a wide percentile)
percentage = 0.05
gain_min, gain_max = symmetric_percentiles(gain_map, percent=percentage)
readnoise_min, readnoise_max = symmetric_percentiles(readnoise_map, percent=percentage)
filter_indices = np.where((gain_map < gain_min) | (gain_map > gain_max) | (readnoise_map < readnoise_min) | (readnoise_map > readnoise_max))
gain_map[filter_indices] = np.nan
readnoise_map[filter_indices] = np.nan

# Save the gain map
np.save(save_to_original_map, gain_map)
print(f"Saved gain map to '{save_to_original_map}'")

# Normalise the gain map to the minimum ISO value
gain_map_normalised = camera.normalise_iso(ISO, gain_map)

# Save the normalised gain map
np.save(save_to_normalised_map, gain_map_normalised)
print(f"Saved normalised gain map to '{save_to_normalised_map}'")
