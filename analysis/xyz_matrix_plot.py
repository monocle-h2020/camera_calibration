"""
Compare a camera's colour space to CIE XYZ and sRGB.
The camera's spectral response functions are plotted together with the xyz colour matching functions,
and its colour space is plotted in xy coordinates.

Command line arguments:
    * `folder`: folder containing the Camera information file.
"""

from sys import argv
from spectacle import spectral, io

# Get the data folder from the command line
folder = io.path_from_input(argv)
root = io.find_root_folder(folder)

# Load Camera objects
camera = io.load_camera(root)
print(f"Loaded Camera object: {camera}")

# Save folder
savefolder_plot = camera.filename_analysis("spectral_response", makefolders=True)

# Load the SRFs
camera._load_spectral_response()

# Plot xyz curves and SRFs
SRF_wavelengths, SRF_RGB = camera.spectral_response[0], camera.spectral_response[1:4]
spectral.plot_xyz_and_rgb(SRF_wavelengths, SRF_RGB, label=camera.name, saveto=savefolder_plot/"SRF_vs_XYZ.pdf")

# Get the camera's colour space base vectors
base_xy = camera.colour_space()

# Plot the colour gamut
spectral.plot_xy_on_gamut(base_xy, label=camera.name, saveto=savefolder_plot/"colour_space.pdf")
