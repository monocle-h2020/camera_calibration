"""
Analyse the covariances between spectral response data at different wavelengths
in a single monochromator run.

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
save_to_SNR = savefolder/f"monochromator_{label}_SNR_cov.pdf"
save_to_cov = savefolder/f"monochromator_{label}_covariance.pdf"
save_to_corr = savefolder/f"monochromator_{label}_correlation.pdf"
save_to_SNR_G = savefolder/f"monochromator_{label}_SNR_cov_G_mean.pdf"
save_to_cov_G = savefolder/f"monochromator_{label}_covariance_G_mean.pdf"

# Find the filenames
mean_files = sorted(folder.glob("*_mean.npy"))

# Blocksize, to slice the arrays with
# This is the block size for RGBG2 data, meaning the mosaicked data will have
# twice this size.
blocksize = 100
center = camera.central_slice(blocksize, blocksize)

# Load all files
splitter = lambda p: float(p.stem.split("_")[0])
wavelengths, means = io.load_means(folder, selection=center, retrieve_value=splitter)

# NaN if a channel's mean value is near saturation
means[means >= 0.95 * camera.saturation] = np.nan

# Bias correction
means = camera.correct_bias(means, selection=center)

# Flat-field correction
means = camera.correct_flatfield(means, selection=center)

# Demosaick the data
means_RGBG2 = np.array(camera.demosaick(*means, selection=center))

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

def cast_back_to_RGBG2(data, indices):
    """
    Cast flattened data back into a (4, N) array.
    """
    data_RGBG2 = [data[ind] for ind in indices]
    return data_RGBG2

# Calculate mean SRF and covariance between all elements
srf = np.nanmean(means_flattened, axis=1)
srf_cov = np.cov(means_flattened)

# Calculate the variance (ignoring covariance) from the diagonal elements
srf_var = np.diag(srf_cov)

# Plot the SRFs with their standard deviations, variance, and SNR
means_plot = cast_back_to_RGBG2(srf, RGBG2)
variance_plot = cast_back_to_RGBG2(srf_var, RGBG2)

spectral.plot_monochromator_curves(wavelengths, means_plot, variance_plot, title=f"{camera.name}: Raw spectral curve ({folder.stem})", unit="ADU", saveto=save_to_SNR)

# Plot the covariances
ticks = [(ind.start + ind.stop) / 2 for ind in RGBG2]
ticklabels = [f"${c}$" for c in ["R", "G", "B", "G_2"]]

plot.plot_covariance_matrix(srf_cov, title=f"Covariances in {folder.stem}", ticks=ticks, ticklabels=ticklabels, saveto=save_to_cov)

# Plot the correlations
srf_correlation = correlation_from_covariance(srf_cov)

plot.plot_covariance_matrix(srf_correlation, title=f"Correlations in {folder.stem}", nr_bins=8, vmin=-1, vmax=1, ticks=ticks, ticklabels=ticklabels, saveto=save_to_corr)

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
srf_G_cov = M_G_G2 @ srf_cov @ M_G_G2.T

srf_G_var = np.diag(srf_G_cov)

# Plot the SRFs with their standard deviations, variance, and SNR
RGB = RGBG2[:3]
means_plot = cast_back_to_RGBG2(srf_G, RGB)
variance_plot = cast_back_to_RGBG2(srf_G_var, RGB)

spectral.plot_monochromator_curves(wavelengths, means_plot, variance_plot, title=f"{camera.name}: Raw spectral curve ({folder.stem})", unit="ADU", saveto=save_to_SNR_G)

# Plot the covariances
ticks, ticklabels = ticks[:3], ticklabels[:3]

plot.plot_covariance_matrix(srf_G_cov, title=f"Covariances in {folder.stem} (mean $G, G_2$)", ticks=ticks, ticklabels=ticklabels, saveto=save_to_cov_G)
