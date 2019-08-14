"""
Create a look-up table for the ISO-gain normalisation function of a camera,
using mean images of the same scene taken at various ISO speeds.

Command line arguments:
    * `folder`: folder containing stacked data for different ISO speeds
"""

import numpy as np
from sys import argv
from spectacle import io, iso, calibrate

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.load_metadata(root)
min_iso = phone["software"]["ISO min"]
max_iso = phone["software"]["ISO max"]
results_iso = results/"iso"
colours      = io.load_colour(stacks)
print("Loaded metadata")

isos, means = io.load_means(folder, retrieve_value=io.split_iso)
isos, stds = io.load_stds(folder, retrieve_value=io.split_iso)
print("Loaded data")

means = calibrate.correct_bias(root, means)

relative_errors = stds / means
median_relative_error = np.median(relative_errors)
print(f"Median relative error in photometry: {median_relative_error*100:.1f} %")

assert isos.min() == min_iso

ratios = means / means[isos.argmin()]
ratios_mean = ratios.mean(axis=(1,2))
ratios_errs = ratios.std (axis=(1,2))

model_type, model, R2, parameters, errors = iso.fit_iso_normalisation_relation(isos, ratios_mean, ratios_errs=ratios_errs, min_iso=min_iso, max_iso=max_iso)

iso_range = np.arange(0, max_iso+1, 1)

lookup_table = np.stack([iso_range, model(iso_range)])
data         = np.stack([isos, ratios_mean, ratios_errs])

np.save(products/"iso_lookup_table.npy", lookup_table)
np.save(products/"iso_data.npy", data)

model_array = np.stack([len(parameters) * [model_type], parameters, errors])
np.savetxt(products/"iso_model.dat", model_array, fmt="%s")
