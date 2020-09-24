"""
Calculate the matrices for converting camera RGB colours to CIE XYZ
coordinates. Both E (equal-energy) and D65 (daylight) illuminants will
be supported.

Command line arguments:
    * `folder`: folder containing the Camera information file.
"""

from sys import argv
from spectacle import xyz, spectral, io

# Get the data folder from the command line
folder = io.path_from_input(argv)
root = io.find_root_folder(folder)

# Load Camera objects
camera = io.load_camera(root)
print(f"Loaded Camera object: {camera}")
