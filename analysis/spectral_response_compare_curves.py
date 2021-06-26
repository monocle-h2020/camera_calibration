"""
Compare two spectral response curves, for example from different methods or
different cameras.

Command line arguments:
    * `files`: two files containing spectral response curves. These should be
    NPY files generated using ../calibration/spectral_response_monochromator.py
    or a similar script.

TO DO:
    * increase the number of input files possible
"""

import numpy as np
from sys import argv
from spectacle import io, plot
from spectacle.general import RMS

# Get the data folder from the command line
files = io.path_from_input(argv)
save_to = io.results_folder/"spectrum_difference.pdf"

# Load the data
curves = [np.loadtxt(f, delimiter=",", unpack=True) for f in files]
print("Loaded data")

# Wavelength grid to interpolate the curves to
wavelength_grid = np.arange(390, 700, 0.5)

def interpolate_curve(curve, wavelengths):
    """
    Interpolate RGBG2 spectral responses to a given `wavelengths` grid
    """
    new_curve = np.tile(np.nan, (len(curve), len(wavelengths)))
    for j, row in enumerate(curve):
        if j == 0:
            continue
        new_curve[j] = np.interp(wavelengths, curve[0], curve[j])
        new_curve[j] = new_curve[j] / np.max(new_curve[j])
    new_curve[0] = wavelengths
    return new_curve

# Interpolate the curves to the same wavelength grid
new_curves = np.array([interpolate_curve(curve, wavelength_grid) for curve in curves])

# Calculate the difference in each filter band
diffs = new_curves[0,1:5] - new_curves[1,1:5]

# Plot the difference
plot.plot_spectrum(wavelength_grid, diffs, ylabel="Difference in relative sensitivity", saveto=save_to)
print(f"Saved plot to '{save_to}'")

# Calculate the RMS difference in total and per band
print(f"RMS difference: {RMS(diffs):.2f}")
print(f"RMS difference in RGBG: {RMS(diffs, axis=1)}")
