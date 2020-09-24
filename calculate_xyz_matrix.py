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

# Load the SRFs
camera._load_spectral_response()
SRF_wavelengths, SRF_RGB = camera.spectral_response[0], camera.spectral_response[1:4]

# Plot xyz curves and SRFs
kwargs = {"lw": 3}
colours = ["#d95f02", "#1b9e77", "#7570b3"]
fig, axs = plt.subplots(nrows=2, figsize=(4,3), sharex=True)
for c, letter, colour in zip([xyz.x, xyz.y, xyz.z], "xyz", colours):
    axs[0].plot(xyz.wavelengths, c, c=colour, label=f"$\\bar {letter}$", **kwargs)
axs[0].set_ylabel("XYZ Response")
axs[0].legend(loc="upper left", bbox_to_anchor=(1,1))
axs[0].set_xlim(390, 700)
axs[0].set_ylim(ymin=0)

for c, letter, colour in zip(SRF_RGB, "rgb", colours):
    axs[1].plot(SRF_wavelengths, c, c=colour, label=f"${letter}$", **kwargs)
axs[1].set_xlabel("Wavelength [nm]")
axs[1].set_ylabel("RGB response")
axs[1].set_ylim(ymin=0)
axs[1].legend(loc="upper left", bbox_to_anchor=(1,1))

plt.show()
plt.close()

