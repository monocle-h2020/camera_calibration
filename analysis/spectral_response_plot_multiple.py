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
from spectacle import io, spectral
from spectacle.general import RMS

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
camera_names = [camera.name for camera in cameras]

# Line styles for the individual camera spectra
styles = ["-", "--", ":", "-."]

# Get the wavelengths from each dataset
wavelengths = [curve[0] for curve in curves]
SRFs_RGBG2 = [curve[1:5] for curve in curves]
SRFs_err_RGBG2 = [curve[5:] for curve in curves]

# Plot the spectral responses in the RGBG2 filters
spectral.plot_spectral_responses(wavelengths, SRFs_RGBG2, labels=camera_names, saveto=save_to_rgbg2)
print(f"Saved RGBG2 plot to '{save_to_rgbg2}'")

# Plot the spectral responses in the RGB filters, with G the mean of G and G2
SRFs_RGB = [spectral.convert_RGBG2_to_RGB(SRF_RGBG2) for SRF_RGBG2 in SRFs_RGBG2]
spectral.plot_spectral_responses(wavelengths, SRFs_RGB, labels=camera_names, saveto=save_to_rgb)
print(f"Saved RGB plot to '{save_to_rgb}'")

# Print the typical differences between G and G2 for each camera
for camera, curve in zip(cameras, SRFs_RGBG2):
    # Calculate and print the RMS difference between G and G2
    print(f"{camera.name:>20} RMS(G-G2) = {RMS(curve[1] - curve[3]):.4f}")

# Plot the signal-to-noise ratio (SNR) in the RGB filters
SRFs_err_RGB = [spectral.convert_RGBG2_to_RGB_uncertainties(SRF_err_RGBG2) for SRF_err_RGBG2 in SRFs_err_RGBG2]
SNRs = [SRF/SRF_err for SRF, SRF_err in zip(SRFs_RGB, SRFs_err_RGB)]
spectral.plot_spectral_responses(wavelengths, SNRs, labels=camera_names, ylabel="Signal-to-noise ratio (SNR)", ylim=(0, None), saveto=save_to_snr)
print(f"Saved SNR plot to '{save_to_snr}'")
