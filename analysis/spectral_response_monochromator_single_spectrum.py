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

# Load the data
wavelengths, *_, means_RGBG2 = spectral.load_monochromator_data(camera, folder, flatfield=True)

# Reshape array
# First remove the spatial information
means_flattened = np.reshape(means_RGBG2, (len(wavelengths), 4, -1))
# Then swap the wavelength and filter axes
means_flattened = np.swapaxes(means_flattened, 0, 1)
# Finally, flatten the array further
means_flattened = np.reshape(means_flattened, (4*len(wavelengths), -1))

# Indices to select R, G, B, and G2
R, G, B, G2 = [np.s_[len(wavelengths)*j : len(wavelengths)*(j+1)] for j in range(4)]
RGBG2 = [R, G, B, G2]

# Calculate mean SRF and covariance between all elements
srf = np.nanmean(means_flattened, axis=1)
srf_cov = np.cov(means_flattened)

# Calculate the variance (ignoring covariance) from the diagonal elements
srf_var = np.diag(srf_cov)

# Plot the SRFs with their standard deviations, variance, and SNR
means_plot = np.reshape(srf, (4,-1))
variance_plot = np.reshape(srf_var, (4,-1))

spectral.plot_monochromator_curves(wavelengths, means_plot, variance_plot, title=f"{camera.name}: Raw spectral curve ({label})", unit="ADU", saveto=save_to_spectrum)

# Plot the covariances
ticks_major = [ind.start for ind in RGBG2] + [RGBG2[-1].stop]
ticks_minor = [(ind.start + ind.stop) / 2 for ind in RGBG2]
ticklabels = [f"${c}$" for c in ["R", "G", "B", "G_2"]]

plot.plot_covariance_matrix(srf_cov, title=f"Covariances in {label}", majorticks=ticks_major, minorticks=ticks_minor, ticklabels=ticklabels, saveto=save_to_covariance)

# Plot the correlations
srf_correlation = correlation_from_covariance(srf_cov)

plot.plot_covariance_matrix(srf_correlation, title=f"Correlations in {label}", nr_bins=8, vmin=-1, vmax=1, majorticks=ticks_major, minorticks=ticks_minor, ticklabels=ticklabels, saveto=save_to_correlation)

# Plot an example
for c, ind in zip("rgby", RGBG2):
    plt.plot(wavelengths, srf_correlation[G,ind][0], c=c)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Correlation")
plt.xlim(wavelengths[0], wavelengths[-1])
plt.grid(ls="--")
plt.show()
plt.close()

# Calculate mean of G and G2
I = np.eye(len(wavelengths))
M_G_G2 = np.zeros((len(wavelengths)*3, len(wavelengths)*4))
M_G_G2[R,R] = I
M_G_G2[B,B] = I
M_G_G2[G,G] = 0.5*I
M_G_G2[G,G2] = 0.5*I

srf_G = M_G_G2 @ srf
srf_cov_G = M_G_G2 @ srf_cov @ M_G_G2.T

srf_var_G = np.diag(srf_cov_G)

# Plot the SRFs with their standard deviations, variance, and SNR
RGB = RGBG2[:3]
means_plot = np.reshape(srf_G, (3,-1))
variance_plot = np.reshape(srf_var_G, (3,-1))

spectral.plot_monochromator_curves(wavelengths, means_plot, variance_plot, title=f"{camera.name}: Raw spectral curve ({label})", unit="ADU", saveto=save_to_spectrum_G)

# Plot the covariances
ticks_major, ticks_minor, ticklabels = ticks_major[:-1], ticks_minor[:3], ticklabels[:3]

plot.plot_covariance_matrix(srf_cov_G, title=f"Covariances in {label} (mean $G, G_2$)", majorticks=ticks_major, minorticks=ticks_minor, ticklabels=ticklabels, saveto=save_to_covariance_G)

# Plot the correlations
srf_correlation_G = correlation_from_covariance(srf_cov_G)

plot.plot_covariance_matrix(srf_correlation_G, title=f"Correlations in {label} (mean $G, G_2$)", label="Correlation", nr_bins=8, vmin=-1, vmax=1, majorticks=ticks_major, minorticks=ticks_minor, ticklabels=ticklabels, saveto=save_to_correlation_G)

# Analyse the difference in correlations between the RGBG2 and RGB data
srf_correlation_without_G2 = srf_correlation[:len(srf_correlation_G),:len(srf_correlation_G)]
srf_correlation_difference = srf_correlation_without_G2 - srf_correlation_G

plot.plot_covariance_matrix(srf_correlation_difference, title=f"Correlations in {label}\nDifferences between RGBG$_2$ and RGB", label="Correlation", nr_bins=8, vmin=-1, vmax=1, majorticks=ticks_major, minorticks=ticks_minor, ticklabels=ticklabels, saveto=save_to_correlation_diff)
