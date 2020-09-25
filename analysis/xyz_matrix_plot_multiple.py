"""
Compare a camera's colour space to CIE XYZ and sRGB.
The camera's spectral response functions are plotted together with the xyz colour matching functions,
and its colour space is plotted in xy coordinates.

Command line arguments:
    * `folder`: any number of folders containing the Camera information files.
"""

from sys import argv
from spectacle import spectral, io

# Get the data folder from the command line
files = io.path_from_input(argv)
roots = [io.find_root_folder(file) for file in files]

# Load Camera objects
cameras = [io.load_camera(root) for root in roots]
print(f"Loaded Camera objects: {cameras}")

# Save folder
savefolder_plot = io.results_folder

# Load the SRFs
for camera in cameras:
    camera._load_spectral_response()

# Plot xyz curves and SRFs
SRF_wavelengths, SRF_RGB = zip(*[(camera.spectral_response[0], camera.spectral_response[1:4]) for camera in cameras])
camera_names = [camera.name for camera in cameras]
spectral.plot_xyz_and_rgb(SRF_wavelengths, SRF_RGB, label=camera_names, saveto=savefolder_plot/"SRF_vs_XYZ.pdf")

# Get the camera's colour space base vectors
base_xy = [camera.colour_space() for camera in cameras]

# Plot the colour gamut
spectral.plot_xy_on_gamut(base_xy, label=camera_names, saveto=savefolder_plot/"colour_space.pdf")
