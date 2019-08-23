"""
Create a gain map using gain images, fitting the mean and standard deviation
against each other. Data for a single ISO speed are loaded and fitted.

Command line arguments:
    * `folder`: folder containing stacked gain data for a single ISO speed
"""

import numpy as np
from sys import argv
from spectacle import io, calibrate

# Get the data folder from the command line
folder = io.path_from_input(argv)
root = io.find_root_folder(folder)

# Get the camera metadata
camera = io.load_metadata(root)
print("Loaded metadata")

# Get the ISO speed of these data from the folder name
ISO = io.split_iso(folder)

# Load the data
names, means = io.load_means(folder)
names, stds = io.load_stds(folder)
print("Loaded data")

# Bias correction
means = calibrate.correct_bias(root, means)

# Use variance instead of standard deviation
variance = stds**2

# Empty arrays to hold the result
gain_map = np.tile(np.nan, means.shape[1:])
readnoise_map = gain_map.copy()

# Loop over the pixels in the array and fit each response individually
fit_max = 0.95 * camera.saturation
for i in range(means.shape[1]):
    for j in range(means.shape[2]):
        m = means[:,i,j] ; v = variance[:,i,j]
        ind = np.where(m < fit_max)
        try:
            gain_map[i,j], readnoise_map[i,j] = np.polyfit(m[ind], v[ind], 1, w=1/m[ind])
        except:
            # Keep a NaN value if fitting is not possible for whatever reason
            pass

    # Progress counter: give percentage done every 42nd row
    if i % 42 == 0:
        print(f"{100 * i / means.shape[1]:.1f}%", end=" ", flush=True)

# Save the gain map
save_to_original_map = root/f"products/gain/gain_iso{ISO}.npy"
np.save(save_to_original_map, gain_map)
print(f"Saved gain map to '{save_to_original_map}'")

# Normalise the gain map to the minimum ISO value
gain_map_normalised = calibrate.normalise_iso(root, gain_map, ISO)

# Save the normalised gain map
save_to_normalised_map = root/f"results/gain_map.npy"
np.save(save_to_normalised_map, gain_map_normalised)
print(f"Saved normalised gain map to '{save_to_normalised_map}'")
