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
from matplotlib import pyplot as plt
from spectacle import spectral, io, plot

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
kwargs = {"lw": 3}
colours = ["#d95f02", "#1b9e77", "#7570b3"]
fig, axs = plt.subplots(nrows=2, figsize=(4,3), sharex=True)
for c, letter, colour in zip(spectral.cie_xyz, "xyz", colours):
    axs[0].plot(spectral.cie_wavelengths, c, c=colour, label=f"$\\bar {letter}$", **kwargs)
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

# Calculate the RGB-to-XYZ matrix
M_RGB_to_XYZ = spectral.calculate_XYZ_matrix(SRF_wavelengths, SRF_RGB)

# Determine the base vectors and their chromaticities
base_xy = spectral.calculate_xy_base_vectors(M_RGB_to_XYZ)

# Plot the colour gamut
plot.plot_xy_on_gamut(base_xy, label=camera.name, saveto=savefolder_plot/"colour_space.pdf")

# Save the conversion matrix
np.savetxt(savefile, M_RGB_to_XYZ, header=f"Matrix for converting {camera.name} RGB data to CIE XYZ, with an equal-energy illuminant (E).")
print(f"Saved XYZ conversion matrix to `{savefile}`.")
