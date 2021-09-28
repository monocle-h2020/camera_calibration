"""
Determine the spectral response curves of a camera based on data from a
monochromator. The data are expected to be in subfolders of a main folder, each
subfolder corresponding to a monochromator setting (e.g. filter/grating).

Command line arguments:
    * `folder`: folder containing subfolders with monochromator data. The sub-
    folders correspond to different monochromator settings (e.g. gratings or
    filters). Each subfolder in turn contains NPY stacks of monochromator data
    taken at different wavelengths.
    * `wvl1`, `wvl2`: the minimum and maximum wavelengths to evaluate at.
    Defaults to 390, 700 nm, respectively, if none are given.
"""

import numpy as np
from sys import argv
from spectacle import io, spectral, plot
from spectacle.general import correlation_from_covariance

# Get the data folder and minimum and maximum wavelengths from the command line
# Defaults
if len(argv) == 2:
    folder = io.path_from_input(argv)
    wvl1 = 390
    wvl2 = 700
elif len(argv) == 4:
    folder, wvl1, wvl2 = io.path_from_input(argv)
    wvl1 = float(wvl1.stem) ; wvl2 = float(wvl2.stem)
else:
    raise ValueError(f"Expected 2 or 4 arguments in `argv`, got {len(argv)}.")

root = io.find_root_folder(folder)

# Load Camera object
camera = io.load_camera(root)
print(f"Loaded Camera object: {camera}")

# Save location based on camera name
save_to_SRF = camera.filename_calibration("spectral_response.csv")
save_to_bands = camera.filename_calibration("spectral_bands.csv")
save_to_XYZ_matrix = camera.filename_calibration("RGB_to_XYZ_matrix.csv")

# Save locations for intermediaries
savefolder = camera.filename_intermediaries("spectral_response", makefolders=True)
save_to_wavelengths = savefolder/"monochromator_wavelengths.npy"
save_to_means = savefolder/"monochromator_raw_means.npy"
save_to_stds = savefolder/"monochromator_raw_stds.npy"
save_to_means_calibrated = savefolder/"monochromator_calibrated_means.npy"
save_to_stds_calibrated = savefolder/"monochromator_calibrated_stds.npy"
save_to_means_normalised = savefolder/"monochromator_normalised_means.npy"
save_to_stds_normalised = savefolder/"monochromator_normalised_stds.npy"
save_to_final_curve = savefolder/"monochromator_curve.npy"

# Get the subfolders in the given data folder
folders = io.find_subfolders(folder)

# Load the data from each subfolder
wavelengths, *_, means_RGBG2, labels = spectral.load_monochromator_data_multiple(camera, folders)
print("Loaded data")

# Find and load the calibration data
cal_files = [sorted(subfolder.glob("*.cal"))[0] for subfolder in folders]
cals = [spectral.load_cal_NERC(file) for file in cal_files]
print("Loaded calibration data")

# Apply the calibration
means_RGBG2 = spectral.apply_calibration_NERC_multiple(cals, wavelengths, means_RGBG2)
print("Applied calibration to data")

# Sort data: longest first, then by amount of overlap with the longest
longest = np.argmax([len(wvl) for wvl in wavelengths])
overlaps = [spectral.wavelength_overlap(wvl, wavelengths[longest]) for wvl in wavelengths]
order = np.argsort(overlaps)[::-1]
means_RGBG2 = [means_RGBG2[i] for i in order]  # Can't do [order] with lists :(
wavelengths = [wavelengths[i] for i in order]

# Reshape array
means_flattened, RGBG2_slices, wavelengths_flattened, labels_flattened = spectral.flatten_monochromator_image_data_multiple(means_RGBG2, wavelengths, labels)
RGBG2_labels = np.tile(plot.RGBG2_latex, len(RGBG2_slices))

# Calculate mean SRF and covariance between all elements
srf = np.nanmean(means_flattened, axis=1)
srf_covariance = np.cov(means_flattened)

# Plot the covariances
ticks_major, ticks_minor = plot.get_tick_locations_from_slices(RGBG2_slices)

plot.plot_covariance_matrix(srf_covariance, title="Covariances", majorticks=ticks_major, minorticks=ticks_minor, ticklabels=RGBG2_labels)

# Plot the correlations
srf_correlation = correlation_from_covariance(srf_covariance)

plot.plot_correlation_matrix(srf_correlation, title="Correlations", majorticks=ticks_major, minorticks=ticks_minor, ticklabels=RGBG2_labels)

nr_bands=4
# Normalise the spectra to each other, then add them all up
while len(wavelengths) > 1:  # As long as multiple data sets are present
    # Find the overlapping wavelengths between this data set and the main one
    wavelengths_overlap, indices_original, indices_new = np.intersect1d(wavelengths[0], wavelengths[1], return_indices=True)

    # Make slices for the appropriate RGBG2 data
    indices_original_RGBG2 = indices_original + (np.arange(nr_bands) * len(wavelengths[0]))[:,np.newaxis]
    indices_new_RGBG2 = nr_bands*len(wavelengths[0]) + indices_new + (np.arange(nr_bands) * len(wavelengths[1]))[:,np.newaxis]

    # Divide the overlapping data element-wise
    ratio = srf[indices_original_RGBG2] / srf[indices_new_RGBG2]

    # Fit a polynomial to those ratios, and apply the same polynomial to the entire data set
    Lambda = np.stack([np.ones_like(wavelengths_overlap), wavelengths_overlap, wavelengths_overlap**2], axis=1)
    # Make all-zero array and then put Lambda into it

    # Take the weighted average of the main data set and this new, normalised one

    # Bookkeeping: remove and rename elements for the next iteration

raise Exception

# Normalise the calibrated data
# Create a copy of the array to put the normalised data into
all_means_normalised = all_means_calibrated.copy()
all_stds_normalised = all_stds_calibrated.copy()

# Loop over the spectra and normalise them by the data set with the largest overlap
for i in normalise_order:
    # Calculate the ratio between this spectrum and the comparison one at each wavelength
    ratios = all_means_calibrated[i] / all_means_normalised[comparison]
    print(f"Normalising spectrum {i} to spectrum {comparison}")

    # Fit a parabolic function to the ratio between the spectra where they overlap
    ind = ~np.isnan(ratios[:,0])
    fits = np.polyfit(all_wavelengths[ind], ratios[ind], 2)
    fit_norms = np.array([np.polyval(f, all_wavelengths) for f in fits.T]).T

    # Normalise by dividing the spectrum by this parabola
    all_means_normalised[i] = all_means_calibrated[i] / fit_norms
    all_stds_normalised[i] = all_stds_calibrated[i] / fit_norms

# Save the normalised curves to file
np.save(save_to_means_normalised, all_means_normalised)
np.save(save_to_stds_normalised, all_stds_normalised)
print(f"Saved normalised curves to {savefolder}")

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
result = np.array(np.stack([all_wavelengths, *response_normalised.T, *errors_normalised.T]))
np.save(save_to_final_curve, result)
print(f"Saved final curves to {savefolder}")

np.savetxt(save_to_SRF, result.T, delimiter=",", header="Wavelength, R, G, B, G2, R_err, G_err, B_err, G2_err")
print(f"Saved spectral response curves to '{save_to_SRF}'")

# Calculate the effective spectral bandwidth of each channel and save those too
bandwidths = spectral.effective_bandwidth(all_wavelengths, response_normalised, axis=0)
np.savetxt(save_to_bands, bandwidths[:,np.newaxis].T, delimiter=", ", header="R, G, B, G2")
print("Effective spectral bandwidths:")
for band, width in zip(plot.RGBG2, bandwidths):
    print(f"{band:<2}: {width:5.1f} nm")

# Calculate the RGB-to-XYZ matrix
M_RGB_to_XYZ = spectral.calculate_XYZ_matrix(all_wavelengths, response_normalised.T)

# Save the conversion matrix
header = f"Matrix for converting {camera.name} RGB data to CIE XYZ, with an equal-energy illuminant (E).\n\
[X_R  X_G  X_B]\n\
[Y_R  Y_G  Y_B]\n\
[Z_R  Z_G  Z_B]"
np.savetxt(save_to_XYZ_matrix, M_RGB_to_XYZ, header=header, fmt="%1.6f", delimiter=", ")
print(f"Saved XYZ conversion matrix to `{save_to_XYZ_matrix}`.")
