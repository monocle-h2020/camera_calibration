"""
This script uses the results obtained using the SPECTACLE method and combines
them into a format that can be uploaded to the database.
"""

from sys import argv
import numpy as np
from spectacle import io
from os import makedirs

# Get the data folder from the command line
folder = io.path_from_input(argv)
root = io.find_root_folder(folder)

# Load metadata
camera = io.load_metadata(root)
print("Loaded metadata")

# Create results folder
identifier = f"{camera.device.manufacturer}-{camera.device.name}"
identifier = identifier.replace(" ", "_")
save_folder = io.results_folder/"spectacle"/identifier

makedirs(save_folder, exist_ok=True)  # create folder if it does not yet exist
print(f"Found/Created save folder '{save_folder}'")

# General properties

generic_header = f"\
SPECTACLE data sheet. More information can be found in our paper (https://doi.org/10.1364/OE.27.019075) and on our website (http://spectacle.ddq.nl/). \n\
Camera manufacturer: {camera.device.manufacturer}\n\
Camera device: {camera.device.name}\n"

# Linearity

# Bias

# Read-out noise

# Dark current

# ISO speed normalization

# Gain variations

# Flat-field correction

# Spectral response
fetch_folder = results/"spectral_response"
try:
    spectral_response = np.load(fetch_folder/"monochromator_curve.npy")
except FileNotFoundError:
    print("No spectral response curve found")
    spectral_response = np.tile(np.nan, (9, 156))

spectral_response = spectral_response.T  # transpose to have columns for wavelength, R, G, ...
header = "This file contains spectral response data. \n\
wavelength (nm),R,G,B,G2,R_error,G_error,B_error,G2_error"

np.savetxt(save_folder/f"spectral_response_{identifier}.csv", spectral_response, delimiter=",", fmt="%.8f", header=generic_header+header)
