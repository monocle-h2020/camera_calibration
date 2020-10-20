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
from spectacle import io, spectral

# Get the data folder from the command line
folder = io.path_from_input(argv)
root = io.find_root_folder(folder)
label = folder.stem

# Load Camera object
camera = io.load_camera(root)
print(f"Loaded Camera object: {camera}")

# Save locations
savefolder = camera.filename_analysis("spectral_response", makefolders=True)
save_to_data = savefolder/f"monochromator_{label}_data.pdf"
save_to_SNR = savefolder/f"monochromator_{label}_SNR.pdf"

# Find the filenames
mean_files = sorted(folder.glob("*_mean.npy"))

# Half-blocksize, to slice the arrays with
blocksize = 100
d = blocksize//2

# Empty arrays to hold the output
wvls  = np.zeros((len(mean_files)))
means = np.zeros((len(mean_files), 4, (blocksize+1)**2))

# Loop over all files
print("Wavelengths [nm]:", end=" ", flush=True)
for j, mean_file in enumerate(mean_files):
    # Load the mean data
    m = np.load(mean_file)

    # Bias correction
    m = camera.correct_bias(m)

    # Demosaick the data
    mean_RGBG = camera.demosaick(m)

    # Select the central blocksize x blocksize pixels
    midx, midy = np.array(mean_RGBG.shape[1:])//2
    sub = mean_RGBG[:,midx-d:midx+d+1,midy-d:midy+d+1]
    sub = sub.reshape(4, -1)

    # NaN if a channel's mean value is near saturation
    sub[sub >= 0.95 * camera.saturation] = np.nan

    # Store results
    means[j] = sub
    wvls[j] = mean_file.stem.split("_")[0]

    print(wvls[j], end=" ", flush=True)

# Reshape array: sort by filter first, then by wavelength
# First len(wvls) elements are R, then G, then B, then G2
means_RGBG2 = np.concatenate([means[:,0], means[:,1], means[:,2], means[:,3]])

# Calculate mean SRF and covariance between all elements
srf = np.nanmean(means_RGBG2, axis=1)
srf_cov = np.cov(means_RGBG2)

# Calculate the variance (ignoring covariance) from the diagonal elements
diag = np.where(np.eye(len(wvls) * 4))
srf_var = srf_cov[diag]
srf_std = np.sqrt(srf_var)
