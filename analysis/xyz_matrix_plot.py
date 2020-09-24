"""
Calculate the matrices for converting camera RGB colours to CIE XYZ
coordinates. Both E (equal-energy) and D65 (daylight) illuminants will
be supported.

Following http://www.ryanjuckett.com/programming/rgb-color-space-conversion/

Command line arguments:
    * `folder`: folder containing the Camera information file.
"""

from sys import argv
import numpy as np
from spectacle import spectral, io

# Get the data folder from the command line
folder = io.path_from_input(argv)
root = io.find_root_folder(folder)

# Load Camera objects
camera = io.load_camera(root)
print(f"Loaded Camera object: {camera}")

# Save folder
savefile = camera.filename_calibration("RGB_to_XYZ_matrix.csv")
savefolder_plot = camera.filename_analysis("spectral_response", makefolders=True)

# Load the SRFs
camera._load_spectral_response()
SRF_wavelengths, SRF_RGB = camera.spectral_response[0], camera.spectral_response[1:4]

# Plot xyz curves and SRFs
spectral.plot_xyz_and_rgb(SRF_wavelengths, SRF_RGB, label=camera.name, saveto=savefolder_plot/"SRF_vs_XYZ.pdf")

# Calculate the RGB-to-XYZ matrix
M_RGB_to_XYZ = spectral.calculate_XYZ_matrix(SRF_wavelengths, SRF_RGB)

# Determine the base vectors and their chromaticities
base_xy = spectral.calculate_xy_base_vectors(M_RGB_to_XYZ)

# Plot the colour gamut
spectral.plot_xy_on_gamut(base_xy, label=camera.name, saveto=savefolder_plot/"colour_space.pdf")

# Save the conversion matrix
header = f"Matrix for converting {camera.name} RGB data to CIE XYZ, with an equal-energy illuminant (E).\n\
[X_R  X_G  X_B]\n\
[Y_R  Y_G  Y_B]\n\
[Z_R  Z_G  Z_B]"
np.savetxt(savefile, M_RGB_to_XYZ, header=header, fmt="%1.6f", delimiter=", ")
print(f"Saved XYZ conversion matrix to `{savefile}`.")
