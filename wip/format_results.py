"""
This script uses the results obtained using the SPECTACLE method and combines
them into a format that can be uploaded to the database.

Please note that this script is still work-in-progress.
"""

from sys import argv
import numpy as np
from spectacle import io
from os import makedirs

# Get the data folder from the command line
folder = io.path_from_input(argv)
root = io.find_root_folder(folder)

calibration = root/"calibration/"
analysis = root/"analysis/"

# Load Camera object
camera = io.load_camera(root)
print(f"Loaded Camera object: {camera}")

# Create results folder
identifier = f"{camera.manufacturer}-{camera.name}"
identifier = identifier.replace(" ", "_")
save_folder = io.results_folder/"spectacle"/identifier

makedirs(save_folder, exist_ok=True)  # create folder if it does not yet exist
print(f"Found/Created save folder '{save_folder}'")

# General properties

generic_header = f"\
SPECTACLE data sheet. More information can be found in our paper (https://doi.org/10.1364/OE.27.019075) and on our website (http://spectacle.ddq.nl/). \n\
Camera manufacturer: {camera.manufacturer}\n\
Camera device: {camera.name}\n"

# Linearity

# Bias

# Read-out noise

# Dark current

# ISO speed normalization

# Gain variations

# Flat-field correction

# Spectral response
# Provided in NPY-format (for use with numpy) and CSV format (for those who do
# not use numpy)
filename_spectral_response = calibration/"spectral_response.npy"
try:
    spectral_response = np.load(filename_spectral_response)
except FileNotFoundError:
    print(f"No spectral response curve found at '{filename_spectral_response}'")
else:
    # Load the data and save it as a CSV
    spectral_response = spectral_response.T  # transpose to have columns for wavelength, R, G, ...
    header_spectral_response = "This file contains spectral response data. \n\
wavelength (nm),R,G,B,G2,R_error,G_error,B_error,G2_error"

    np.savetxt(save_folder/f"spectral_response_{identifier}.csv", spectral_response, delimiter=",", fmt="%.8f", header=generic_header+header_spectral_response)

    # Copy the NPY-format data as well
