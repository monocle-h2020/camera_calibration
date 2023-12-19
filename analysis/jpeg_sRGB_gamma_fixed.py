"""
Compare JPEG linearity data to sRGB models with given gamma factors and
calculate the best-fitting normalisation as well as the R^2 value of the best-
fitting model and the RMS difference between model and data.

These calculations are all done for each individual pixel.

Command line arguments:
    * `folder`: the folder containing linearity data stacks. These should be
    NPY stacks taken at different exposure conditions, with the same ISO speed.
    * `gamma`: the gamma value of the sRGB model to compare to the data. Any
    number of gamma values can be provided.
"""

import numpy as np
from sys import argv
from spectacle import io, linearity as lin

# Get the data folder from the command line
folder, *gammas = io.path_from_input(argv)
gammas = [float(str(gamma)) for gamma in gammas]
root = io.find_root_folder(folder)

# Load Camera object
camera = io.load_camera(root)
print(f"Loaded Camera object: {camera}")

# Save locations
savefolder = camera.filename_intermediaries("jpeg", makefolders=True)

# Load the data
intensities_with_errors, jmeans = io.load_jmeans(folder, retrieve_value=lin.filename_to_intensity)
intensities, intensity_errors = intensities_with_errors.T
print("Loaded data")

# Loop over the given gamma values
for gamma in gammas:
    # Fit the model
    print(f"\nFitting sRGB model with gamma = {gamma} ...")
    normalisations, Rsquares, RMSes, RMSes_relative = lin.sRGB_compare_gamma(intensities, jmeans, gamma=gamma)

    # Combine the results into a single array and save it to file
    result_combined = np.stack([normalisations, Rsquares, RMSes, RMSes_relative])
    save_to = savefolder/f"sRGB_comparison_gamma{gamma}.npy"
    np.save(save_to, result_combined)
    print(f"Saved results to '{save_to}'")
