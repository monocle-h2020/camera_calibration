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
root, images, stacks, products, results = io.folders(folder)

# Get the camera metadata
phone = io.load_metadata(root)
fit_max = 0.95 * 2**phone["camera"]["bits"]
colours = io.load_colour(stacks)

# Get the ISO speed of these data from the folder name
ISO = io.split_iso(folder)
print("Loaded metadata")

# Load the data
names, means = io.load_means (folder  )
names, stds  = io.load_stds  (folder  )
print("Loaded data")

means = calibrate.correct_bias(root, means)

variance = stds**2

gains = np.tile(np.nan, means.shape[1:])
rons  = gains.copy()

for i in range(means.shape[1]):
    for j in range(means.shape[2]):
        m = means[:,i,j] ; v = variance[:,i,j]
        ind = np.where(m < fit_max)
        try:
            gains[i,j], rons[i,j] = np.polyfit(m[ind], v[ind], 1, w=1/m[ind])
        except:
            pass

    if i%15:
        print(f"{100 * i / means.shape[1]:.1f}%", end=" ", flush=True)

np.save(root/f"products/gain/iso{ISO}.npy", gains)
