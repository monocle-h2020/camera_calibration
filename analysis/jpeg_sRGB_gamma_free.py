"""
Compare JPEG linearity data to sRGB models with free gamma factors and find the
best-fitting gamma and normalisation as well as the R^2 value of the best-
fitting model.

These calculations are all done for each individual pixel.

Command line arguments:
    * `folder`: the folder containing linearity data stacks. These should be
    NPY stacks taken at different exposure conditions, with the same ISO speed.
"""

import numpy as np
from sys import argv
from spectacle import io, linearity as lin

# Get the data folder from the command line
folder = io.path_from_input(argv)
root = io.find_root_folder(folder)

# Load Camera object
camera = io.load_camera(root)
print(f"Loaded Camera object: {camera}")

# Save locations
savefolder = camera.filename_intermediaries("flatfield", makefolders=True)
save_to_result = savefolder/"sRGB_model_free.npy"

# Load the data
intensities_with_errors, jmeans = io.load_jmeans(folder, retrieve_value=lin.filename_to_intensity)
intensities, intensity_errors = intensities_with_errors.T
print("Loaded data")

# Fit the model
print("Fitting sRGB model with free gamma...")
normalisations, gammas, R2s = lin.fit_sRGB_generic(intensities, jmeans)

# Combine the results into a single array and save it to file
result_combined = np.stack([normalisations, gammas, R2s])
np.save(save_to_result, result_combined)
print(f"Saved results to '{save_to_result}'")
