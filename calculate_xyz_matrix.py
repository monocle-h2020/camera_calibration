"""
Calculate the matrices for converting camera RGB colours to CIE XYZ
coordinates. Both E (equal-energy) and D65 (daylight) illuminants will
be supported.

Command line arguments:
    * `folder`: folder containing the Camera information file.
"""

from sys import argv
import numpy as np
from matplotlib import pyplot as plt
from spectacle import xyz, spectral, io

# Get the data folder from the command line
folder = io.path_from_input(argv)
root = io.find_root_folder(folder)

# Load Camera objects
camera = io.load_camera(root)
print(f"Loaded Camera object: {camera}")

# Plot xyz curves
for c, letter in zip([xyz.x, xyz.y, xyz.z], "xyz"):
    plt.plot(xyz.wavelengths, c, label=f"$\\bar {letter}$")
plt.xlabel("Wavelength [nm]")
plt.ylabel("Response")
plt.legend(loc="best")
plt.xlim(390, 700)
plt.ylim(ymin=0)
plt.show()
