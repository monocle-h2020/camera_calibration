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

plot.plot_correlation_matrix_diagonal_multi(srf_correlation, RGBG2_slices, wavelengths)
plot.plot_correlation_matrix_diagonal_RGBG2_multi(srf_correlation, RGBG2_slices, wavelengths)

nr_bands=4
polynomial_degree=2
# Normalise the spectra to each other, then add them all up
while len(wavelengths) > 1:  # As long as multiple data sets are present
    # Transfer matrix: 1 everywhere for now. Parts will be changed as we go.
    M = np.eye(len(srf))

    # Slices corresponding to the band being normalised
    start_band1 = len(wavelengths[0]) * nr_bands
    edges_band1_RGBG2 = start_band1 + np.arange(5) * len(wavelengths[1])
    slices_band1_RGBG2 = [slice(start, stop) for start, stop in zip(edges_band1_RGBG2, edges_band1_RGBG2[1:])]

    # Find the overlapping wavelengths between this data set and the main one
    wavelengths_overlap, indices_original, indices_new = np.intersect1d(wavelengths[0], wavelengths[1], return_indices=True)
    single_overlap_length = len(wavelengths_overlap)
    full_overlap_length = nr_bands * single_overlap_length

    # Make index lists for the appropriate RGBG2 data
    indices_original_RGBG2 = indices_original + (np.arange(nr_bands) * len(wavelengths[0]))[:,np.newaxis]
    indices_new_RGBG2 = nr_bands*len(wavelengths[0]) + indices_new + (np.arange(nr_bands) * len(wavelengths[1]))[:,np.newaxis]

    ### Divide the overlapping data element-wise
    ratio = srf[indices_original_RGBG2] / srf[indices_new_RGBG2]
    ratio_flattened = np.ravel(ratio)

    # Approximation of division using the Jacobian matrix; for propagating the covariance.
    J_ratio = np.zeros((M.shape[0] + full_overlap_length, M.shape[0]))
    J_ratio[:-full_overlap_length] = np.eye(len(srf))

    # Ratios for the Jacobian matrix
    dr_dyA = 1 / srf[indices_new_RGBG2]
    dr_dyB = -ratio / srf[indices_new_RGBG2]

    # Indices for the Jacobian matrix
    indices_goal = M.shape[0] + np.arange(nr_bands+1) * single_overlap_length
    slices_goal = [slice(start, stop) for start, stop in zip(indices_goal, indices_goal[1:])]

    # Insert these elements into J_ratio
    for s_goal, ind_original, ind_new, JA, JB in zip(slices_goal, indices_original_RGBG2, indices_new_RGBG2, dr_dyA, dr_dyB):
        J_ratio[s_goal,ind_original] = np.diag(JA)
        J_ratio[s_goal,ind_new] = np.diag(JB)

    srf_covariance_with_ratio = J_ratio @ srf_covariance @ J_ratio.T

    # Plot the resulting correlation matrix, just to be sure
    srf_correlation_with_ratio = correlation_from_covariance(srf_covariance_with_ratio)
    plot.plot_correlation_matrix(srf_correlation_with_ratio, title="Correlations -- Ratio between data sets", majorticks=ticks_major, minorticks=ticks_minor, ticklabels=RGBG2_labels, vmin=-0.25, vmax=0.25)

    ### Fit a polynomial to those ratios, and apply the same polynomial to the entire data set
    Lambda_single = np.stack([np.ones_like(wavelengths_overlap), wavelengths_overlap, wavelengths_overlap**2], axis=1)  # Polynomial base array, for the overlap in one band
    Lambda_bands = np.vstack([Lambda_single]*nr_bands)  # Polynomial coefficients for the overlap, repeated for each band

    # For simplicity, we put nr_bands copies of Lambda_single into an otherwise all-zero array so it can be multiplied directly with srf and the covariance.
    # This could also be done using clever indexing, but appending zeros makes it easier to read.
    Lambda_output = np.stack([np.ones_like(wavelengths[1]), wavelengths[1], wavelengths[1]**2], axis=1)  # Polynomial coefficients corresponding to all wavelengths in wavelengths[1] - for applying the fit
    Lambda_term = Lambda_output @ np.linalg.inv(Lambda_bands.T @ Lambda_bands) @ Lambda_bands.T
    ratio_fitted = Lambda_term @ ratio_flattened

    # Jacobian matrix for fitting the ratio
    J_ratio_fit = np.zeros((M.shape[0] + len(wavelengths[1]), srf_covariance_with_ratio.shape[0]))
    J_ratio_fit[:M.shape[0], :M.shape[0]] = np.eye(len(srf))
    J_ratio_fit[M.shape[0]:, M.shape[0]:] = Lambda_term

    srf_covariance_with_ratio_fit = J_ratio_fit @ srf_covariance_with_ratio @ J_ratio_fit.T

    # Plot the resulting correlation matrix, just to be sure
    srf_correlation_with_ratio_fit = correlation_from_covariance(srf_covariance_with_ratio_fit)
    plot.plot_correlation_matrix(srf_correlation_with_ratio_fit, title="Correlations -- Fitted ratio", majorticks=ticks_major, minorticks=ticks_minor, ticklabels=RGBG2_labels, vmin=-0.25, vmax=0.25)

    ### Apply the fitted ratio to the original data
    # Make the full transfer matrix, which is 1 everywhere outside band 1
    for s in slices_band1_RGBG2:
        M[s,s] = np.diag(ratio_fitted)
    srf_normalised = M @ srf

    # Jacobian matrix for applying the fitted ratio
    slice_ratio_fitted = slice(len(srf), len(srf)+len(ratio_fitted))
    J_normalise = np.zeros((len(srf), srf_covariance_with_ratio_fit.shape[0]))
    J_normalise[:len(srf), :len(srf)] = np.eye(len(srf))  # 1 everywhere except ratios
    for s in slices_band1_RGBG2:
        J_normalise[s,s] = np.diag(ratio_fitted)  # dynew / dy_old = ratio_fitted
        J_normalise[s,slice_ratio_fitted] = np.diag(srf[s])  # dynew / dr = y_old

    srf_covariance_normalised = J_normalise @ srf_covariance_with_ratio_fit @ J_normalise.T

    # Plot the resulting correlation matrix, just to be sure
    srf_correlation_normalised = correlation_from_covariance(srf_covariance_normalised)
    plot.plot_correlation_matrix(srf_correlation_normalised, title="Correlations -- Normalised", majorticks=ticks_major, minorticks=ticks_minor, ticklabels=RGBG2_labels)

    ### Average of the two data sets
    wavelengths_combined = np.unique([*wavelengths[0], *wavelengths[1]])
    size_new_and_old = nr_bands * (len(wavelengths[0]) + len(wavelengths[1]))
    size_combined = nr_bands * len(wavelengths_combined)
    size_full = len(wavelengths_flattened) - nr_bands * (len(wavelengths[0]) + len(wavelengths[1])) + size_combined
    size_remaining = size_full - size_combined

    # Get the overlap between the new data and the old data 0, old data 1, and overlap
    _, indices_goal_original, indices_original_goal = np.intersect1d(wavelengths_combined, wavelengths[0], return_indices=True)
    _, indices_goal_new, indices_new_goal = np.intersect1d(wavelengths_combined, wavelengths[1], return_indices=True)
    indices_new_goal += nr_bands*len(wavelengths[0])
    _, indices_goal_overlap, _ = np.intersect1d(wavelengths_combined, wavelengths_overlap, return_indices=True)

    indices_goal_original_RGBG2 = indices_goal_original + (np.arange(nr_bands) * len(wavelengths_combined))[:,np.newaxis]
    indices_original_goal_RGBG2 = indices_original_goal + (np.arange(nr_bands) * len(wavelengths[0]))[:,np.newaxis]

    indices_goal_new_RGBG2 = indices_goal_new + (np.arange(nr_bands) * len(wavelengths_combined))[:,np.newaxis]
    indices_new_goal_RGBG2 = indices_new_goal + (np.arange(nr_bands) * len(wavelengths[1]))[:,np.newaxis]

    indices_goal_overlap_RGBG2 = indices_goal_overlap + (np.arange(nr_bands) * len(wavelengths_combined))[:,np.newaxis]

    # Start with an empty array
    M_weighted_average = np.zeros((size_full, len(srf_normalised)))

    # Fill with 1s for the overlap between the old/new and combined data
    M_weighted_average[indices_goal_original_RGBG2, indices_original_goal_RGBG2] = 1.
    M_weighted_average[indices_goal_new_RGBG2, indices_new_goal_RGBG2] = 1.

    # Put 1/2 (averaging) where both old and new data are used; this overwrites some 1s
    M_weighted_average[indices_goal_overlap_RGBG2,indices_original_RGBG2] = 0.5
    M_weighted_average[indices_goal_overlap_RGBG2,indices_new_RGBG2] = 0.5

    # The rest of the matrix, corresponding to the other data sets, should be kept as the identity
    if size_remaining:
        M_weighted_average[-size_remaining:, -size_remaining:] = np.eye(size_remaining)

    # Finally, apply the weighted average to the SRF and covariance
    srf_averaged = M_weighted_average @ srf_normalised
    srf_covariance_averaged = M_weighted_average @ srf_covariance_normalised @ M_weighted_average.T

    ### Bookkeeping: remove and rename elements for the next iteration
    _ = wavelengths.pop(1)  # Remove the wavelengths for the new data set
    wavelengths[0] = wavelengths_combined  # Replace the wavelengths for the old data set with the combined data set
    wavelengths_flattened = spectral.flatten_image_properties(wavelengths)[0]
    srf = srf_averaged
    srf_covariance = srf_covariance_averaged

    RGBG2_slices = [spectral.generate_slices_for_RGBG2_bands(len(w)) for w in wavelengths]
    RGBG2_slices = spectral.adjust_slices_for_RGBG2_bands_multi(RGBG2_slices)
    RGBG2_labels = np.tile(plot.RGBG2_latex, len(RGBG2_slices))
    ticks_major, ticks_minor = plot.get_tick_locations_from_slices(RGBG2_slices)

    # Plot the result
    srf_correlation = correlation_from_covariance(srf_covariance)
    plot.plot_correlation_matrix(srf_correlation, title="Correlations -- Averaged", majorticks=ticks_major, minorticks=ticks_minor, ticklabels=RGBG2_labels)

# Tidy up the result
wavelengths = wavelengths[0]
RGBG2_slices = RGBG2_slices[0]
variance = np.diagonal(srf_covariance)

# Plot results
plot.plot_correlation_matrix(srf_correlation, title="Correlations after combining data", majorticks=ticks_major, minorticks=ticks_minor, ticklabels=RGBG2_labels)

plot.plot_correlation_matrix_diagonal(srf_correlation, RGBG2_slices, wavelengths)
plot.plot_correlation_matrix_diagonal_RGBG2(srf_correlation, RGBG2_slices, wavelengths)

# Plot the resulting SRF
srf_split = np.reshape(srf, (4, -1))
variance_split = np.reshape(variance, (4, -1))
spectral.plot_monochromator_curves(wavelengths, srf_split, variance_split, title=f"{camera.name}: Combined spectral curve", unit="ADU")

raise Exception

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
