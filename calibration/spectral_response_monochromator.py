"""
Determine the spectral response curves of a camera based on data from a
monochromator. The data are expected to be in subfolders of a main folder, each
subfolder corresponding to a monochromator setting (e.g. filter/grating).

Command line arguments:
    * `folder`: folder containing monochromator data (in subfolders)
"""

import numpy as np
from sys import argv
from spectacle import io, spectral

# Get the data folder and minimum and maximum wavelengths from the command line
folder, wvl1, wvl2 = io.path_from_input(argv)
wvl1 = float(wvl1.stem) ; wvl2 = float(wvl2.stem)
root = io.find_root_folder(folder)

# Get the camera metadata
camera = io.load_metadata(root)
print("Loaded metadata")

# Get the subfolders in the given data folder
folders = io.find_subfolders(folder)

# Load the data from each subfolder
spectra = [spectral.load_monochromator_data(root, subfolder) for subfolder in folders]
print("Loaded data")

# Find and load the calibration data
cal_files = [sorted(subfolder.glob("*.cal"))[0] for subfolder in folders]
cals = [spectral.load_cal_NERC(file) for file in cal_files]
print("Loaded calibration data")

# Combine the spectral data from each folder into the same format
all_wvl = np.unique(np.concatenate([spec[:,0] for spec in spectra]))
all_means = np.tile(np.nan, (len(spectra), len(all_wvl), 4))
all_stds = all_means.copy()

# Add the data from the separate spectra into one big array
# If a spectrum is missing a wavelength, keep that value NaN
# Note: data may be missing at lower or higher wavelengths, but not within the
# spectrum itself. This is TO DO.
for i, spec in enumerate(spectra):
    min_wvl, max_wvl = spec[:,0].min(), spec[:,0].max()
    min_in_all = np.where(all_wvl == min_wvl)[0][0]
    max_in_all = np.where(all_wvl == max_wvl)[0][0]
    all_means[i][min_in_all:max_in_all+1] = spec[:,1:5]
    all_stds[i][min_in_all:max_in_all+1] = spec[:,5:]

# Save the raw curves to file
np.save(root/"intermediaries/spectral_response/monochromator_wavelengths.npy", all_wvl)
np.save(root/"intermediaries/spectral_response/monochromator_raw_means.npy", all_means)
np.save(root/"intermediaries/spectral_response/monochromator_raw_stds.npy", all_stds)
print(f"Saved raw curves to '{root/'intermediaries/spectral_response/'}'")

# Calibrate the data
# First, create a copy of the array to put the calibrated data into
all_means_calibrated = all_means.copy()
all_means_calibrated[:] = np.nan
all_stds_calibrated = all_means_calibrated.copy()

# Loop over the spectra
for i, (mean, std, cal) in enumerate(zip(all_means, all_stds, cals)):
    # Create an empty (NaN) copy of the data, to store the result in
    calibrated = mean.copy() ; calibrated[:] = np.nan

    # Find the overlapping wavelengths between calibration and data
    overlap, cal_indices, all_wvl_indices = np.intersect1d(cal[0], all_wvl, return_indices=True)

    # Calibrate the data and store it in the main array
    calibrated[all_wvl_indices] = mean[all_wvl_indices] / cal[1, cal_indices, np.newaxis]
    all_means_calibrated[i] = calibrated

    # Assume the error in the result is dominated by the error in the data,
    # not in the calibration (strong assumption!) and propagate the error
    calibrated_std = std.copy() ; calibrated_std[:] = np.nan
    calibrated_std[all_wvl_indices] = std[all_wvl_indices] / cal[1, cal_indices, np.newaxis]
    all_stds_calibrated[i] = calibrated_std

# Save the calibrated curves to file
np.save(root/"intermediaries/spectral_response/monochromator_calibrated_means.npy", all_means_calibrated)
np.save(root/"intermediaries/spectral_response/monochromator_calibrated_stds.npy", all_stds_calibrated)
print(f"Saved calibrated curves to '{root/'intermediaries/spectral_response/'}'")

# Normalise the calibrated data
# Create a copy of the array to put the normalised data into
all_means_normalised = all_means_calibrated.copy()
all_stds_normalised = all_stds_calibrated.copy()

def overlap(spectrum_a, spectrum_b):
    """
    Find the number of overlapping wavelengths between two spectra
    """
    summed = spectrum_a + spectrum_b
    length = len(np.where(~np.isnan(summed))[0])
    return length

# Find the number of overlapping wavelengths
all_overlaps = np.array([[overlap(mean1, mean2) for mean1 in all_means_calibrated] for mean2 in all_means_calibrated])

# Use the spectrum with the most overlap with itself (i.e. most data) as the
# baseline to normalise others to
baseline = np.diag(all_overlaps).argmax()

# The order to normalise in, based on the amount of overlap with the baseline
# -2 because there is no point in normalising the baseline (-1) by itself
normalise_order = np.argsort(all_overlaps[baseline])[-2::-1]

# Loop over the spectra and normalise them by the data set with the largest
# overlap
for i in normalise_order:
    # If there is any overlap with the baseline, normalise to that
    if all_overlaps[i,baseline]:
        comparison = baseline
    # If not, normalise to another spectrum
    else:
        # NB should add a check to make sure the `comparison` is one that has
        # previously been normalised itself
        # Alternately do two iterations?
        comparison = np.argsort(all_overlaps[i])[-2]

    # Calculate the ratio between this spectrum and the comparison one at each
    # wavelength
    ratios = all_means_calibrated[i] / all_means_normalised[comparison]
    print(f"Normalising spectrum {i} to spectrum {comparison}")

    # Fit a parabolic function to the ratio between the spectra where they
    # overlap
    ind = ~np.isnan(ratios[:,0])
    fits = np.polyfit(all_wvl[ind], ratios[ind], 2)
    fit_norms = np.array([np.polyval(f, all_wvl) for f in fits.T]).T

    # Normalise by dividing the spectrum by this parabola
    all_means_normalised[i] = all_means_calibrated[i] / fit_norms
    all_stds_normalised[i] = all_stds_calibrated[i] / fit_norms

# Save the normalised curves to file
np.save(root/"intermediaries/spectral_response/monochromator_normalised_means.npy", all_means_normalised)
np.save(root/"intermediaries/spectral_response/monochromator_normalised_stds.npy", all_stds_normalised)
print(f"Saved normalised curves to '{root/'intermediaries/spectral_response/'}'")

# Combine the spectra into one
# Calculate the signal-to-noise ratio (SNR) at each wavelength in each spectrum
SNR = all_means_normalised / all_stds_normalised

# Mask NaN data
mean_mask = np.ma.array(all_means_normalised, mask=np.isnan(all_means_normalised))
stds_mask = np.ma.array(all_stds_normalised , mask=np.isnan(all_stds_normalised ))
SNR_mask  = np.ma.array(SNR                 , mask=np.isnan(SNR                 ))

# Calculate the weight of each spectrum at each wavelength, based on the SNR
weights = SNR_mask**2

# Calculate the weighted average (and its error) per wavelength
flat_means_mask = np.ma.average(mean_mask, axis=0, weights=weights)
flat_errs_mask = np.sqrt(np.ma.sum((weights/weights.sum(axis=0) * stds_mask)**2, axis=0))

# Calculate the SNR of the resulting spectrum
SNR_final = flat_means_mask / flat_errs_mask

# Remove the mask from the final data set
response_normalised = (flat_means_mask / flat_means_mask.max()).data
errors_normalised = (flat_errs_mask / flat_means_mask.max()).data

# Combine the result into one big array and save it
result = np.array(np.stack([all_wvl, *response_normalised.T, *errors_normalised.T]))
np.save(root/"intermediaries/spectral_response/monochromator_curve.npy", result)
print(f"Saved final curves to '{root/'intermediaries/spectral_response/'}'")

save_to = root/"calibration/spectral_response.npy"
np.save(save_to, result)
print(f"Saved spectral response curves to '{save_to}'")

# Calculate the effective spectral bandwidth of each channel and save those too
bandwidths = spectral.effective_bandwidth(all_wvl, response_normalised, axis=0)
np.savetxt(root/"calibration/spectral_bandwidths.dat", bandwidths)
print("Effective spectral bandwidths:")
for band, width in zip([*"RGB", "G2"], bandwidths):
    print(f"{band:<2}: {width:5.1f} nm")
