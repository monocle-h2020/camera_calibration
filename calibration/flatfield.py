"""
Create a flat-field map using the mean flat-field images.

Command line arguments:
    * `meanfile`: location of an NPY stack of mean flat-field data. It is
    assumed that for a meanfile "X_mean.npy", a standard deviation stack can be
    found at "X_stds.npy" in the same folder.

To do:
    * Save map as simply `flat_field.npy` or with a label depending on user
    input.
"""

import numpy as np
from sys import argv
from spectacle import io, flat
from spectacle.general import gaussMd, correlation_from_covariance, uncertainty_from_covariance
from matplotlib import pyplot as plt

# Get the data folder from the command line
meanfile = io.path_from_input(argv)
root = io.find_root_folder(meanfile)
label = meanfile.stem.split("_mean")[0]

# Load Camera object
camera = io.load_camera(root)
print(f"Loaded Camera object: {camera}")

# Replace the calibration file? TO DO: make this a command line argument
overwrite_calibration = True

# Save locations
savefolder = camera.filename_intermediaries("flatfield", makefolders=True)
save_to_correction = savefolder/f"flatfield_correction_{label}.npy"
save_to_correction_raw = savefolder/f"flatfield_correction_{label}_raw.npy"
save_to_correction_modelled_intermediary = savefolder/f"flatfield_correction_{label}_modelled.npy"
save_to_parameters_intermediary = savefolder/f"flatfield_parameters_{label}.csv"

# Save location based on camera name
save_to_parameters_calibration = camera.filename_calibration("flatfield_parameters.csv")

# Load the data
stdsfile = meanfile.parent / meanfile.name.replace("mean", "stds")
mean = np.load(meanfile)
stds = np.load(stdsfile)
print("Loaded data")

# Bias correction
mean = camera.correct_bias(mean)

# Normalise the RGBG2 channels to a maximum of 1 each
mean_normalised, stds_normalised = flat.normalise_RGBG2(mean, stds, camera.bayer_map)
print("Normalised data")

# Convolve the flat-field data with a Gaussian kernel to remove small-scale variations
flatfield_gauss = gaussMd(mean_normalised, 10)

# Calculate the correction factor
correction = 1 / flatfield_gauss
correction_raw = 1 / mean_normalised

# Save the correction factor maps
np.save(save_to_correction, correction)
np.save(save_to_correction_raw, correction_raw)
print(f"Saved the flat-field correction maps to '{save_to_correction}' (Gaussed) and '{save_to_correction_raw}' (raw)")

# Only use the inner X pixels
correction_clipped = flat.clip_data(correction)
correction_raw_clipped = flat.clip_data(correction_raw)

# Fit a radial vignetting model
print("Fitting...")
parameters, covariance = flat.fit_vignette_radial(correction_clipped)
uncertainties = uncertainty_from_covariance(covariance)
correlation = correlation_from_covariance(covariance)

# Output the best-fitting model parameters and errors
print("Parameter +- Uncertainty ; Relative uncertainty")
for p, s in zip(parameters, uncertainties):
    print(f"{p:+.6f} +- {s:.6f}    ; {abs(100*s/p):.3f} %")

# Plot the correlation matrix
plt.imshow(correlation, cmap=plt.cm.get_cmap("cividis", 8), aspect="equal", origin="lower", vmin=-1, vmax=1)
plt.colorbar()
plt.xlabel("Parameters") ; plt.ylabel("Parameters")
plt.xticks(np.arange(7), flat.parameter_labels_latex)
plt.yticks(np.arange(7), flat.parameter_labels_latex)
plt.title("Correlation matrix")
plt.show()
plt.close()

# Save the best-fitting model parameters
result_array = np.array([*parameters, *uncertainties])[:,np.newaxis].T
save_locations = [save_to_parameters_intermediary, save_to_parameters_calibration] if overwrite_calibration else [save_to_parameters_intermediary]
for saveto in save_locations:
    np.savetxt(saveto, result_array, header=", ".join(flat.parameter_labels + flat.parameter_error_labels), delimiter=",")
    print(f"Saved best-fitting model parameters to '{saveto}'")

# Apply the best-fitting model to the data to generate a correction map
correction_modelled = flat.apply_vignette_radial(correction.shape, parameters)

# Save the moddelled correction map
np.save(save_to_correction_modelled_intermediary, correction_modelled)
print(f"Saved the modelled flat-field correction map to '{save_to_correction_modelled_intermediary}'")
