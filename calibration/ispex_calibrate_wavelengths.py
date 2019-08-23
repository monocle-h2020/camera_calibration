import numpy as np
from sys import argv
from spectacle import general, io, plot, wavelength, raw2

file = io.path_from_input(argv)
root = io.find_root_folder(file)

img = io.load_raw_file(file)
image_cut  = img.raw_image [760:1470, 2150:3900]
colors_cut = img.raw_colors[760:1470, 2150:3900]
x = np.arange(2150, 3900)
y = np.arange(760 , 1470)

RGB = raw2.pull_apart2(image_cut, colors_cut)
plot.show_RGBG(RGB)

RGB_gauss = general.gauss_nan(RGB, sigma=(0,0,10))
plot.show_RGBG(RGB_gauss)

lines = wavelength.find_fluorescent_lines(RGB_gauss) + x[0]
lines_fit = wavelength.fit_fluorescent_lines(lines, y)
plot.plot_fluorescent_lines(y, lines, lines_fit)

wavelength_fits = wavelength.fit_many_wavelength_relations(y, lines_fit)
coefficients, coefficients_fit = wavelength.fit_wavelength_coefficients(y, wavelength_fits)
plot.wavelength_coefficients(y, wavelength_fits, coefficients_fit)
wavelength.save_coefficients(coefficients, saveto=results/"ispex/wavelength_solution.npy")

wavelengths_cut = wavelength.calculate_wavelengths(coefficients, x, y)
wavelengths_split,_ = raw2.pull_apart(wavelengths_cut, colors_cut)
RGBG,_ = raw2.pull_apart(image_cut, colors_cut)

lambdarange, all_interpolated = wavelength.interpolate_multi(wavelengths_split, RGBG)

stacked = wavelength.stack(lambdarange, all_interpolated)
plot.plot_fluorescent_spectrum(stacked[0], stacked[1:])
