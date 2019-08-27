"""
Plot a single spectrum generated with a monochromator, e.g. from a single
grating/filter setting.

This script is intended as a quick check of data quality in case the main
monochromator processing scripts do not work or produce unexpected results.

Command line arguments:
    * `folder`: folder containing monochromator data
"""

import numpy as np
from sys import argv
from spectacle import io, spectral

# Get the data folder from the command line
folder = io.path_from_input(argv)
root = io.find_root_folder(folder)
label = folder.stem
save_folder = root/"analysis/spectral_response/"

# Get the camera metadata
camera = io.load_metadata(root)
print("Loaded metadata")

# Load the data
spectrum = spectral.load_monochromator_data(root, folder)
print("Loaded data")

# Split the spectrum into its constituents
wavelengths = spectrum[:,0]
mean = spectrum[:,1:5]
stds = spectrum[:,5:]

# Plot the raw spectrum
spectral.plot_monochromator_curves(wavelengths, [mean], [stds], title=f"{camera.device.name}: Raw spectral curve ({label})", unit="ADU", saveto=save_folder/f"monochromator_{label}_data.pdf")
print("Saved raw spectrum plot")

# Calculate the signal-to-noise ratio (SNR) and plot it
SNR = mean / stds
SNR_err = np.zeros_like(SNR)  # don't plot errors on the SNR

spectral.plot_monochromator_curves(wavelengths, [SNR], [SNR_err], title=f"{camera.device.name}: Signal-to-noise ratio ({label})", unit="SNR", saveto=save_folder/f"monochromator_{label}_SNR.pdf")
print("Saved SNR plot")
