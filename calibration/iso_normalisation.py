"""
Create a look-up table for the ISO-gain normalisation function of a camera, using mean images of the same scene taken at various ISO speeds.

A camera settings file containing the minimum and maximum ISO speeds is necessary for this script to work.

Command line arguments:
    * `folder`: folder containing NPY stacks of identical exposures taken at different ISO speeds.

Example:
    python calibration/iso_normalisation.py ~/SPECTACLE_data/iPhone_SE/stacks/iso_normalisation/
"""
import numpy as np
from spectacle import io, iso

# Command-line arguments
import argparse
parser = argparse.ArgumentParser(description="Create a look-up table for the ISO-gain normalisation function of a camera.")
parser.add_argument("folder", help="Folder containing .npy stacks of identical exposures taken at different ISO speeds.", type=io.Path)
parser.add_argument("-o", "--output_folder", help="Folder to save results files to (default: camera outputs folder).", type=io.Path, default=None)
parser.add_argument("-v", "--verbose", help="Enable verbose output.", action="store_true")
args = parser.parse_args()

# Get the root data folder
root = io.find_root_folder(args.folder)

# Load Camera object
camera = io.load_camera(root)
if args.verbose:
    print(f"Loaded Camera object: {camera}")
assert hasattr(camera, "settings"), f"A settings file could not be loaded for the following Camera object:\n{camera}"

# Save location based on camera name
filename_model = "iso_normalisation_model.csv"
filename_lookup_table = "iso_normalisation_lookup_table.csv"
save_to_model = camera.filename_calibration(filename_model) if args.output_folder is None else args.output_folder / filename_model
save_to_lookup_table = camera.filename_calibration(filename_lookup_table) if args.output_folder is None else args.output_folder / filename_lookup_table
save_to_data = camera.filename_intermediaries("iso_normalisation/iso_data.npy", makefolders=True)

# Load the mean and standard deviation stacks for each ISO value
isos, means = io.load_means(args.folder, retrieve_value=io.split_iso, leave_progressbar=args.verbose)
isos, stds = io.load_stds(args.folder, retrieve_value=io.split_iso, leave_progressbar=args.verbose)
if args.verbose:
    print("Loaded data")

# Bias correction
means = camera.correct_bias(means)
if args.verbose:
    print("Applied bias correction")

# Get relative errors to use as weights in the fit
relative_errors = stds / means
median_relative_error = np.median(relative_errors)
if args.verbose:
    print(f"Median relative error in photometry: {median_relative_error*100:.1f} %")

# Check that we have data at the lowest ISO speed for this camera
assert isos.min() == camera.settings.ISO_min, f"Lowest ISO speed in the data ({isos.min()}) is greater than the minimum ISO speed available on this camera ({camera.settings.ISO_min})."

# Convert the mean values at each ISO to normalised units, compared to the lowest ISO speed
ratios = means / means[isos.argmin()]
ratios_mean = ratios.mean(axis=(1, 2))
ratios_errs = ratios.std(axis=(1, 2))
print(f"Normalised data to minimum ISO ({camera.settings.ISO_min})")

# Fit a model to the ISO normalisation curve
model_type, model, R2, parameters, errors = iso.fit_iso_normalisation_relation(isos, ratios_mean, ratios_errs=ratios_errs, min_iso=camera.settings.ISO_min, max_iso=camera.settings.ISO_max)

# Save the observed mean normalisation factor at each ISO speed, so it can be
# compared to the model later
data = np.stack([isos, ratios_mean, ratios_errs])
np.save(save_to_data, data)
print(f"\nSaved normalisation data to '{save_to_data}'")

# Save the best-fitting model parameters and their errors
iso.save_iso_model(save_to_model, model_type, parameters, errors)
print(f"Saved model parameters to '{save_to_model}'")

# Apply the best-fitting model to the full ISO range of this camera to create
# a look-up table, then save it
iso_range = np.arange(0, camera.settings.ISO_max+1, 1)
lookup_table = np.stack([iso_range, model(iso_range)]).T
np.savetxt(save_to_lookup_table, lookup_table, header="ISO, Normalisation", fmt="%i, %.6f")
print(f"Saved look-up table to '{save_to_lookup_table}'")
