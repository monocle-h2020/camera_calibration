"""
This script uses the results obtained using the SPECTACLE method and combines
them into a format that can be uploaded to the database.
"""

from sys import argv
import numpy as np
from phonecal import io

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")

# General properties

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

