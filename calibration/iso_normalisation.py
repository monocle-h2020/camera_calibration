"""
Create a look-up table for the ISO-gain normalisation function of a camera,
using mean images of the same scene taken at various ISO speeds.

A bias correction is applied to the data. If available, a bias map is used for
this; otherwise, a mean value from metadata.

Command line arguments:
    * `folder`: folder containing NPY stacks of identical exposures taken at
    different ISO speeds.
"""

import numpy as np
from sys import argv
from spectacle import io, iso, calibrate

# Get the data folder from the command line
folder = io.path_from_input(argv)
root = io.find_root_folder(folder)
save_to_data = root/"intermediaries/iso_normalisation/iso_data.npy"
save_to_model = root/"calibration/iso_normalisation_model.csv"
save_to_lookup_table = root/"calibration/iso_normalisation_lookup_table.csv"

# Get metadata
camera = io.load_metadata(root)
print("Loaded metadata")

# Load the mean and standard deviation stacks for each ISO value
isos, means = io.load_means(folder, retrieve_value=io.split_iso)
isos, stds = io.load_stds(folder, retrieve_value=io.split_iso)
print("Loaded data")

# Bias correction
means = calibrate.correct_bias(root, means)

# Get relative errors to use as weights in the fit
relative_errors = stds / means
median_relative_error = np.median(relative_errors)
print(f"Median relative error in photometry: {median_relative_error*100:.1f} %")

# Check that we have data at the lowest ISO speed for this camera
assert isos.min() == camera.settings.ISO_min, f"Lowest ISO speed in the data ({isos.min()}) is greater than the minimum ISO speed available on this camera ({camera.settings.ISO_min})."

# Convert the mean values at each ISO to normalised units, compared to the
# lowest ISO speed
ratios = means / means[isos.argmin()]
ratios_mean = ratios.mean(axis=(1,2))
ratios_errs = ratios.std (axis=(1,2))
print(f"Normalised data to minimum ISO ({camera.settings.ISO_min})")

# Fit a model to the ISO normalisation curve
model_type, model, R2, parameters, errors = iso.fit_iso_normalisation_relation(isos, ratios_mean, ratios_errs=ratios_errs, min_iso=camera.settings.ISO_min, max_iso=camera.settings.ISO_max)

# Save the observed mean normalisation factor at each ISO speed, so it can be
# compared to the model later
data = np.stack([isos, ratios_mean, ratios_errs])
np.save(save_to_data, data)
print(f"Saved normalisation data to '{save_to_data}'")

# Save the best-fitting model parameters and their errors
iso.save_iso_model(save_to_model, model_type, parameters, errors)
print(f"Saved model parameters to '{save_to_model}'")

# Apply the best-fitting model to the full ISO range of this camera to create
# a look-up table, then save it
iso_range = np.arange(0, camera.settings.ISO_max+1, 1)
lookup_table = np.stack([iso_range, model(iso_range)]).T
np.savetxt(save_to_lookup_table, lookup_table, header="ISO, Normalisation", fmt="%i, %.6f")
print(f"Saved look-up table to '{save_to_lookup_table}'")
