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

# Load first file to make a slice object
mean0 = np.load(mean_files[0])
midx, midy = np.array(mean0.shape)//2
center = np.s_[midx-blocksize:midx+blocksize, midy-blocksize:midy+blocksize]

# Load all files
splitter = lambda p: float(p.stem.split("_")[0])
wvls, means = io.load_means(folder, selection=center, retrieve_value=splitter)

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
means_flattened = np.reshape(means_RGBG2, (len(wvls), 4, -1))
# Then swap the wavelength and filter axes
means_flattened = np.swapaxes(means_flattened, 0, 1)
# Finally, flatten the array further
means_flattened = np.reshape(means_flattened, (4*len(wvls), -1))

# Indices to select R, G, B, and G2
R, G, B, G2 = [np.s_[len(wvls)*j : len(wvls)*(j+1)] for j in range(4)]
RGBG2 = [R, G, B, G2]

# Calculate mean SRF and covariance between all elements
srf = np.nanmean(means_flattened, axis=1)
srf_cov = np.cov(means_flattened)

# Calculate the variance (ignoring covariance) from the diagonal elements
srf_var = np.diag(srf_cov)
srf_std = np.sqrt(srf_var)

# Plot the SRFs with their standard deviations, variance, and SNR
labels = ["Response\n[ADU]", "Variance\n[ADU$^2$]", "SNR"]
fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(4,4))
for ind, c in zip(RGBG2, "rybg"):
    axs[0].plot(wvls, srf[ind], c=c)
    axs[0].fill_between(wvls, srf[ind]-srf_std[ind], srf[ind]+srf_std[ind], color=c, alpha=0.3)

    axs[1].plot(wvls, srf_var[ind], c=c)

    axs[2].plot(wvls, srf[ind]/srf_std[ind], c=c)

for ax, label in zip(axs, labels):
    ax.set_ylabel(label)
    ax.set_ylim(ymin=0)

axs[-1].set_xlim(wvls[0], wvls[-1])
axs[-1].set_xlabel("Wavelength [nm]")

axs[0].set_title(folder.stem)

plt.savefig(save_to_SNR, bbox_inches="tight")
plt.show()
plt.close()

# Plot the covariances
ticks = [(ind.start + ind.stop) / 2 for ind in RGBG2]
ticklabels = [f"${c}$" for c in ["R", "G", "B", "G_2"]]

plot.plot_covariance_matrix(srf_cov, title=f"Covariances in {folder.stem}", ticks=ticks, ticklabels=ticklabels, saveto=save_to_cov)

# Plot the correlations
srf_correlation = correlation_from_covariance(srf_cov)

plot.plot_covariance_matrix(srf_correlation, title=f"Correlations in {folder.stem}", nr_bins=8, vmin=-1, vmax=1, ticks=ticks, ticklabels=ticklabels, saveto=save_to_corr)

# Plot an example
for c, ind in zip("rgby", RGBG2):
    plt.plot(wvls, srf_correlation[G,ind][0], c=c)
plt.xlabel("Wavelength [nm]")
plt.ylabel("Correlation")
plt.xlim(wvls[0], wvls[-1])
plt.grid(ls="--")
plt.show()
plt.close()

# Calculate mean of G and G2
I = np.eye(len(wvls))
M_G_G2 = np.zeros((len(wvls)*3, len(wvls)*4))
M_G_G2[R,R] = I
M_G_G2[B,B] = I
M_G_G2[G,G] = 0.5*I
M_G_G2[G,G2] = 0.5*I

srf_G = M_G_G2 @ srf
srf_G_cov = M_G_G2 @ srf_cov @ M_G_G2.T

srf_G_var = np.diag(srf_G_cov)
srf_G_std = np.sqrt(srf_G_var)

# Plot the SRFs with their standard deviations, variance, and SNR
labels = ["Response\n[ADU]", "Variance\n[ADU$^2$]", "SNR"]
fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(4,4))
for ind, c in zip(RGBG2[:3], "rgb"):
    axs[0].plot(wvls, srf_G[ind], c=c)
    axs[0].fill_between(wvls, srf_G[ind]-srf_G_std[ind], srf_G[ind]+srf_G_std[ind], color=c, alpha=0.3)

    axs[1].plot(wvls, srf_G_var[ind], c=c)

    axs[2].plot(wvls, srf_G[ind]/srf_G_std[ind], c=c)

for ax, label in zip(axs, labels):
    ax.set_ylabel(label)
    ax.set_ylim(ymin=0)

axs[-1].set_xlim(wvls[0], wvls[-1])
axs[-1].set_xlabel("Wavelength [nm]")

axs[0].set_title(folder.stem)

plt.savefig(save_to_SNR_G, bbox_inches="tight")
plt.show()
plt.close()

# Plot the covariances
ticks, ticklabels = ticks[:3], ticklabels[:3]

plot.plot_covariance_matrix(srf_G_cov, title=f"Covariances in {folder.stem} (mean $G, G_2$)", ticks=ticks, ticklabels=ticklabels, saveto=save_to_cov_G)
