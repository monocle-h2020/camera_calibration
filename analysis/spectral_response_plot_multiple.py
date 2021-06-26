"""
Plot multiple spectral response curves, from different cameras and/or methods.

Command line arguments:
    * `files`: up to four files containing spectral response curves. These
    should be NPY files generated using
    ../calibration/spectral_response_monochromator.py or a similar script.
    (more than four files may be given, but only the first four will be shown)

TO DO:
    * increase the number of input files possible
"""

import numpy as np
from sys import argv
from spectacle import io, plot, spectral
from spectacle.general import RMS
from matplotlib import pyplot as plt

# Get the data folder from the command line
files = io.path_from_input(argv)
roots = [io.find_root_folder(file) for file in files]

save_to_rgbg2 = io.results_folder/"spectral_responses.pdf"
save_to_rgb = io.results_folder/"spectral_responses_RGB.pdf"
save_to_snr = io.results_folder/"spectral_responses_SNR.pdf"

# Load Camera objects
cameras = [io.load_camera(root) for root in roots]
print(f"Loaded Camera objects: {cameras}")

# Load the data
curves = [np.loadtxt(f, delimiter=",", unpack=True) for f in files]
print("Loaded data")

# Check that all necessary data are available
assert len(cameras) == len(curves)

number_of_cameras = len(cameras)

# Line styles for the individual camera spectra
styles = ["-", "--", ":", "-."]

# Plot the spectral responses in the RGBG2 filters
# Create a figure to hold the plot
plt.figure(figsize=(7,3), tight_layout=True)

# Loop over the response curves
for i, (curve, camera, style) in enumerate(zip(curves, cameras, styles)):
    wavelength = curve[0]
    plot._rgbgplot(wavelength, curve[1:5], ls=style)

    # Add an invisible line for the legend
    plt.plot([-1000,-1001], [-1000,-1001], c='k', ls=style, label=camera.name)

# Plot parameters
plt.grid(True)
plt.xticks(np.arange(0,1000,50))
plt.xlim(390, 700)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Relative sensitivity")
plt.ylim(0, 1.02)
plt.legend(loc="best")
plt.savefig(save_to_rgbg2)
plt.close()
print(f"Saved RGBG2 plot to '{save_to_rgbg2}'")

# Plot the spectral responses in the RGB filters, with G the mean of G and G2
# Create a figure to hold the plot
plt.figure(figsize=(7,3), tight_layout=True)

# Loop over the response curves
for i, (curve, camera, style) in enumerate(zip(curves, cameras, styles)):
    wavelength = curve[0]

    # Calculate and print the RMS difference between G and G2
    print(f"{camera.name:>15} RMS(G-G2) = {RMS(curve[2] - curve[4]):.4f}")

    # Combine G and G2 into a single curve
    means_RGB = spectral.convert_RGBG2_to_RGB(curve[1:5])

    plot._rgbplot(wavelength, means_RGB, ls=style)

    # Add an invisible line for the legend
    plt.plot([-1000,-1001], [-1000,-1001], c='k', ls=style, label=camera.name)

# Plot parameters
plt.grid(True)
plt.xticks(np.arange(0,1000,50))
plt.xlim(390, 700)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Relative sensitivity")
plt.ylim(0, 1.02)
plt.legend(loc="best")
plt.savefig(save_to_rgb)
plt.close()
print(f"Saved RGB plot to '{save_to_rgb}'")

# Plot the signal-to-noise ratio (SNR) in the RGB filters, with G the mean of G
# and G2
# Create a figure to hold the plot
plt.figure(figsize=(7,3), tight_layout=True)

# Loop over the response curves
for i, (curve, camera, style) in enumerate(zip(curves, cameras, styles)):
    wavelength = curve[0]
    errors = curve[5:]

    # Combine G and G2 into a single curve
    means_RGB = spectral.convert_RGBG2_to_RGB(curve[1:5])
    G_errors = 0.5 * np.sqrt((errors[1::2]**2).sum(axis=0))
    errors_RGB = np.stack([errors[0], G_errors, errors[2]])

    SNR = means_RGB / errors_RGB

    plot._rgbplot(wavelength, SNR, ls=style)

    # Add an invisible line for the legend
    plt.plot([-1000,-1001], [-1000,-1001], c='k',ls=style, label=camera.name)

# Plot parameters
plt.grid(True)
plt.xticks(np.arange(0,1000,50))
plt.xlim(390, 700)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Signal-to-noise ratio (SNR)")
plt.ylim(ymin=0)
plt.legend(loc="best")
plt.savefig(save_to_snr)
plt.close()
print(f"Saved SNR plot to '{save_to_snr}'")
