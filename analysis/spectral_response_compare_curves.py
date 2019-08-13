import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, plot
from spectacle.general import RMS

files = io.path_from_input(argv)

folders = [io.folders(file)[0] for file in files]
cameras = [io.load_metadata(folder)["device"]["name"] for folder in folders]

curves = [np.load(f) for f in files]

wavelength_grid = np.arange(390, 700, 0.5)

def interpolate_curve(curve, wavelengths):
    new_curve = np.tile(np.nan, (len(curve), len(wavelengths)))
    for j, row in enumerate(curve):
        if j == 0:
            continue
        new_curve[j] = np.interp(wavelengths, curve[0], curve[j])
        new_curve[j] = new_curve[j] / np.max(new_curve[j])
    new_curve[0] = wavelengths
    return new_curve

new_curves = np.array([interpolate_curve(curve, wavelength_grid) for curve in curves])
diffs = new_curves[0,1:5] - new_curves[1,1:5]

plot.plot_spectrum(wavelength_grid, diffs, ylabel="Difference in relative sensitivity")

print(f"RMS difference: {RMS(diffs):.2f}")
print(f"RMS difference in RGBG: {RMS(diffs, axis=1)}")
