"""
Analyse a single set of spectral response data from a monochromator,
e.g. from a single grating/filter setting.

Command line arguments:
    * `folder`: folder containing monochromator data. This should contain NPY
    stacks of monochromator data taken at different wavelengths with a single
    settings (e.g. filter/grating).
"""

import numpy as np
from matplotlib import pyplot as plt
from sys import argv
from scipy.linalg import block_diag
from spectacle import io, spectral, plot
from spectacle.general import correlation_from_covariance

# Get the data folder from the command line
folder = io.path_from_input(argv)
root = io.find_root_folder(folder)
label = folder.stem

# Load Camera object
camera = io.load_camera(root)
print(f"Loaded Camera object: {camera}")

# Save locations
savefolder = camera.filename_analysis("spectral_response", makefolders=True)
save_to_spectrum = savefolder/f"monochromator_{label}_spectrum.pdf"
save_to_covariance = savefolder/f"monochromator_{label}_covariance.pdf"
save_to_correlation = savefolder/f"monochromator_{label}_correlation.pdf"
save_to_spectrum_G = savefolder/f"monochromator_{label}_spectrum_RGB.pdf"
save_to_covariance_G = savefolder/f"monochromator_{label}_covariance_RGB.pdf"
save_to_correlation_G = savefolder/f"monochromator_{label}_correlation_RGB.pdf"
save_to_correlation_diff = savefolder/f"monochromator_{label}_correlation_difference.pdf"
save_to_correlation_interp = savefolder/f"monochromator_{label}_correlation_interpolated.pdf"
save_to_spectrum_interp = savefolder/f"monochromator_{label}_spectrum_interpolated.pdf"

# Load the data
wavelengths, *_, means_RGBG2 = spectral.load_monochromator_data(camera, folder, flatfield=True)

# Reshape array
means_flattened, RGBG2_slices = spectral.flatten_monochromator_image_data(means_RGBG2)

# Calculate mean SRF and covariance between all elements
srf = np.nanmean(means_flattened, axis=1)
srf_covariance = np.cov(means_flattened)

# Calculate the variance (ignoring covariance) from the diagonal elements
srf_variance = np.diag(srf_covariance)

# Plot the SRFs with their standard deviations, variance, and SNR
means_plot = np.reshape(srf, (4,-1))
variance_plot = np.reshape(srf_variance, (4,-1))

spectral.plot_monochromator_curves(wavelengths, means_plot, variance_plot, title=f"{camera.name}: Raw spectral curve ({label})", unit="ADU", saveto=save_to_spectrum)

# Plot the covariances
ticks_major, ticks_minor = plot.get_tick_locations_from_slices(RGBG2_slices)

plot.plot_covariance_matrix(srf_covariance, title=f"Covariances in {label}", majorticks=ticks_major, minorticks=ticks_minor, ticklabels=plot.RGBG2_latex, saveto=save_to_covariance)

# Plot the correlations
srf_correlation = correlation_from_covariance(srf_covariance)

plot.plot_correlation_matrix(srf_correlation, title=f"Correlations in {label}", nr_bins=8, majorticks=ticks_major, minorticks=ticks_minor, ticklabels=plot.RGBG2_latex, saveto=save_to_correlation)

# Calculate mean of G and G2
R, G, B, G2 = RGBG2_slices
I = np.eye(len(wavelengths))
M_G_G2 = np.zeros((len(wavelengths)*3, len(wavelengths)*4))
M_G_G2[R,R] = I
M_G_G2[B,B] = I
M_G_G2[G,G] = 0.5*I
M_G_G2[G,G2] = 0.5*I

srf_G = M_G_G2 @ srf
srf_covariance_G = M_G_G2 @ srf_covariance @ M_G_G2.T

srf_variance_G = np.diag(srf_covariance_G)

# Plot the SRFs with their standard deviations, variance, and SNR
means_plot = np.reshape(srf_G, (3,-1))
variance_plot = np.reshape(srf_variance_G, (3,-1))

spectral.plot_monochromator_curves(wavelengths, means_plot, variance_plot, title=f"{camera.name}: Raw spectral curve ({label})", unit="ADU", saveto=save_to_spectrum_G)

# Plot the covariances
RGB_slices = RGBG2_slices[:3]
ticks_major, ticks_minor = plot.get_tick_locations_from_slices(RGB_slices)

plot.plot_covariance_matrix(srf_covariance_G, title=f"Covariances in {label} (mean $G, G_2$)", majorticks=ticks_major, minorticks=ticks_minor, ticklabels=plot.RGB_latex, saveto=save_to_covariance_G)

# Plot the correlations
srf_correlation_G = correlation_from_covariance(srf_covariance_G)

plot.plot_correlation_matrix(srf_correlation_G, title=f"Correlations in {label} (mean $G, G_2$)", nr_bins=8, majorticks=ticks_major, minorticks=ticks_minor, ticklabels=plot.RGB_latex, saveto=save_to_correlation_G)

# Analyse the difference in correlations between the RGBG2 and RGB data
srf_correlation_without_G2 = srf_correlation[:len(srf_correlation_G),:len(srf_correlation_G)]
srf_correlation_difference = srf_correlation_without_G2 - srf_correlation_G

plot.plot_correlation_matrix(srf_correlation_difference, title=f"Correlations in {label}\nDifferences between RGBG$_2$ and RGB", nr_bins=8, majorticks=ticks_major, minorticks=ticks_minor, ticklabels=plot.RGB_latex, saveto=save_to_correlation_diff)

# Linear interpolation
step = 0.5
wavelengths_new = np.arange(wavelengths[0], wavelengths[-1]+step, step)
M = spectral.linear_interpolation_matrix(wavelengths_new, wavelengths)

# Stack copies of B to match the spectral bands
M = spectral.repeat_matrix(M, 4)

# Perform the interpolation
srf_interpolated, covariance_interpolated = spectral.apply_interpolation_matrix(M, srf, srf_covariance)
correlation_interpolated = correlation_from_covariance(covariance_interpolated)

# Plot the results
RGBG2_slices = spectral.generate_slices_for_RGBG2_bands(len(wavelengths_new), 4)
ticks_major, ticks_minor = plot.get_tick_locations_from_slices(RGBG2_slices)

plot.plot_correlation_matrix(correlation_interpolated, title=f"Correlations in {label} (after interpolation)", nr_bins=8, majorticks=ticks_major, minorticks=ticks_minor, ticklabels=plot.RGBG2_latex, saveto=save_to_correlation_interp)

# Calculate the variance (ignoring covariance) from the diagonal elements
variance_interpolated = np.diag(covariance_interpolated)

# Plot the SRFs with their standard deviations, variance, and SNR
means_plot = np.reshape(srf_interpolated, (4,-1))
variance_plot = np.reshape(variance_interpolated, (4,-1))

spectral.plot_monochromator_curves(wavelengths_new, means_plot, variance_plot, title=f"{camera.name}: Interpolated spectral curve ({label})", unit="ADU", saveto=save_to_spectrum_interp)

# Integrate the result
M_integration = spectral.trapezoid_matrix(wavelengths)
M_integration = spectral.repeat_matrix(M_integration, 4)

srf_integral, covariance_integral = spectral.apply_interpolation_matrix(M_integration, srf, srf_covariance)
correlation_integral = correlation_from_covariance(covariance_integral)
