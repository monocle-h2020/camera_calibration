"""
Plot a single spectrum generated with a monochromator, e.g. from a single
grating/filter setting.

This script is intended as a quick check of data quality in case the main
monochromator processing scripts do not work or produce unexpected results.

Command line arguments:
    * `folder`: folder containing monochromator data. This should contain NPY
    stacks of monochromator data taken at different wavelengths with a single
    settings (e.g. filter/grating).
"""
from sys import argv
from spectacle import io, spectral

# Get the data folder from the command line
folder = io.path_from_input(argv)
root = io.find_root_folder(folder)
label = folder.stem

# Load Camera object
camera = io.load_camera(root)
print(f"Loaded Camera object: {camera}")

# Save locations
savefolder = camera.filename_analysis("spectral_response", makefolders=True)
save_to_data = savefolder/f"monochromator_{label}_data.pdf"

# Load the data
wavelengths, mean, stds, _ = spectral.load_monochromator_data(camera, folder)
print("Loaded data")

# Transpose for the plotting function
mean = mean.T
stds = stds.T
variance = stds**2

# Plot the raw spectrum
spectral.plot_monochromator_curves(wavelengths, mean, variance, title=f"{camera.name}: Raw spectral curve ({label})", unit="ADU", saveto=save_to_data)
print("Saved spectrum plot")
