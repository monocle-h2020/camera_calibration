"""
Compare JPEG linearity data to sRGB models with free gamma factors and find the
best-fitting gamma and normalisation as well as the R^2 value of the best-
fitting model.

These calculations are all done for each individual pixel.

Command line arguments:
    * `folder`: the folder containing linearity data stacks.
"""

import numpy as np
from sys import argv
from spectacle import io, linearity as lin

# Get the data folder from the command line
folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
save_folder = root/"results/linearity/"

# Load the data
intensities_with_errors, jmeans = io.load_jmeans(folder, retrieve_value=lin.filename_to_intensity)
intensities, intensity_errors = intensities_with_errors.T
print("Loaded data")

# Fit the model
print("Fitting sRGB model with free gamma...")
normalisations, gammas, R2s = lin.fit_sRGB_generic(intensities, jmeans)

# Save the output
for param, label_simple in zip([normalisations, gammas, R2s], ["normalization", "gamma", "R2"]):
    save_to = save_folder/f"sRGB_free_{label_simple}.npy"
    np.save(save_to, param)
    print(f"Saved {label_simple} results to '{save_to}'")
