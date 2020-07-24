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
from spectacle import io
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
curves = [np.load(f) for f in files]
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

    # Loop over the RGBG2 responses, plotting G in yellow and G2 in green
    for j, c in enumerate("rybg"):
        mean  = curve[1+j]
        error = curve[5+j]

        # Plot the curve
        plt.plot(wavelength, mean, c=c, ls=style)

    # Add an invisible line for the legend
    plt.plot([-1000,-1001], [-1000,-1001], c='k', ls=style, label=camera.device.name)

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
    means = curve[1:5]
    errors = curve[5:]

    # Calculate and print the RMS difference between G and G2
    print(f"{camera.device.name:>15} RMS(G-G2) = {RMS(curve[2] - curve[4]):.4f}")

    # Combine G and G2 into a single curve
    G = means[1::2].mean(axis=0)
    G_errors = 0.5 * np.sqrt((errors[1::2]**2).sum(axis=0))
    means_RGB = np.stack([means[0], G, means[2]])
    errors_RGB = np.stack([errors[0], G_errors, errors[2]])

    # Loop over the RGB responses
    for j, c in enumerate("rgb"):
        mean  =  means_RGB[j]
        error = errors_RGB[j]

        # Plot the curve
        plt.plot(wavelength, mean, c=c, ls=style)

    # Add an invisible line for the legend
    plt.plot([-1000,-1001], [-1000,-1001], c='k',ls=style, label=camera.device.name)

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
    means = curve[1:5]
    errors = curve[5:]

    # Combine G and G2 into a single curve
    G = means[1::2].mean(axis=0)
    G_errors = 0.5 * np.sqrt((errors[1::2]**2).sum(axis=0))
    means_RGB = np.stack([means[0], G, means[2]])
    errors_RGB = np.stack([errors[0], G_errors, errors[2]])

    SNR = means_RGB / errors_RGB

    # Loop over the RGB responses
    for j, c in enumerate("rgb"):
        snr = SNR[j]

        # Plot the curve
        plt.plot(wavelength, snr, c=c, ls=style)

    # Add an invisible line for the legend
    plt.plot([-1000,-1001], [-1000,-1001], c='k',ls=style, label=camera.device.name)

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
