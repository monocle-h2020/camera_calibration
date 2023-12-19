"""
Calibrate the wavelength response of an iSPEX unit using a spectrum of a
fluorescent light.

Command line arguments:
    * `file`: location of a RAW photograph of a fluorescent light spectrum,
    taken with iSPEX.

This should either be made generic (for any spectrometric data) or be forked
into the iSPEX repository.

NOTE: May not function correctly due to changes to flat-fielding methods. This
will be fixed with the general overhaul for iSPEX 2.
"""

import numpy as np
from sys import argv
from spectacle import general, io, plot, wavelength, raw, raw2, flat

# Get the data folder from the command line
file = io.path_from_input(argv)
root = io.find_root_folder(file)
save_to = root/"intermediaries/spectral_response/ispex_wavelength_solution.npy"

# Load Camera object
camera = io.load_camera(root)
print(f"Loaded Camera object: {camera}")

# Load the data
img = io.load_raw_file(file)
print("Loaded data")

# Bias correction
values = camera.correct_bias(img.raw_image)

# Flat-field correction (includes clipping data)
values = camera.correct_flatfield(values)

# Clip the Bayer map to be the same shape
bayer_map = img.raw_colors

# Cut out the spectrum
# Note that these limits are hard-coded
xmin, xmax = 1900, 3500
ymin, ymax = 510, 1220
cut = np.s_[ymin:ymax, xmin:xmax]
image_cut  = values[cut]
colors_cut = bayer_map[cut]
x = np.arange(xmin, xmax)
y = np.arange(ymin, ymax)

# Split the image into RBG (not RGBG2!) using the new pull_apart method
RGB = raw2.pull_apart2(image_cut, colors_cut)
plot.show_RGBG(RGB)

# Convolve the data with a Gaussian kernel on the wavelength axis to remove
# noise and fill in the gaps
RGB_gauss = general.gauss_filter_multidimensional(RGB, sigma=(0,0,10))

# Show the Gaussed image
plot.show_RGBG(RGB_gauss)

# Find the locations of the line peaks in every row
lines = wavelength.find_fluorescent_lines(RGB_gauss) + xmin

# Fit a parabola (smile) to the found positions
lines_fit = wavelength.fit_fluorescent_lines(lines, y)

# Plot the observed and fitted line positions
plot.plot_fluorescent_lines(y, lines, lines_fit)

# Fit a wavelength relation for each row
wavelength_fits = wavelength.fit_many_wavelength_relations(y, lines_fit)

# Fit a polynomial to the coefficients of the previous fit
coefficients, coefficients_fit = wavelength.fit_wavelength_coefficients(y, wavelength_fits)

# Save the coefficients to file
wavelength.save_coefficients(coefficients, saveto=save_to)
print(f"Saved wavelength coefficients to '{save_to}'")

# Convert the input spectrum to wavelengths and plot it, as a sanity check
wavelengths_cut = wavelength.calculate_wavelengths(coefficients, x, y)
wavelengths_split = raw.demosaick(colors_cut, wavelengths_cut)
RGBG = raw.demosaick(colors_cut, image_cut)

lambdarange, all_interpolated = wavelength.interpolate_multi(wavelengths_split, RGBG)

stacked = wavelength.stack(lambdarange, all_interpolated)
plot.plot_fluorescent_spectrum(stacked[0], stacked[1:])
