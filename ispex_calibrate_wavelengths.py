from matplotlib import pyplot as plt
import numpy as np
from sys import argv
from phonecal import general, io, plot, wavelength, raw2

file = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(file)

img = io.load_dng_raw(file)
image_cut  = img.raw_image [760:1470, 2150:3900]
colors_cut = img.raw_colors[760:1470, 2150:3900]
x = np.arange(2150, 3900)
y = np.arange(760 , 1470)

RGB = raw2.pull_apart2(image_cut, colors_cut)
plot.show_RGBG(RGB)

RGB_gauss = general.gauss_nan(RGB, sigma=(0,0,10))
plot.show_RGBG(RGB_gauss)

lines = wavelength.find_fluorescent_lines(RGB_gauss)
lines_fit = wavelength.fit_fluorescent_lines(lines)
plot.fluorescent_lines(y, lines, lines_fit)

wavelength_fits = wavelength.fit_many_wavelength_relations(y, lines_fit)
coefficients, coefficients_fit = wavelength.fit_wavelength_coefficients(y, wavelength_fits)
plot.wavelength_coefficients(y, wavelength_fits, coefficients_fit)
wavelength.save_coefficients(coefficients, saveto=results/"ispex/wavelength_solution.npy")

raise Exception

coefficients = wavelength.load_coefficients("wavelength_solution.npy")

wavelengths_cut = wavelength.calculate_wavelengths(coefficients, raw.x, raw.y)

wavelengths_split, offsets = raw.pull_apart(wavelengths_cut, colors_cut)

lambdarange, all_interpolated = wavelength.interpolate_multi(wavelengths_split, RGBG)
plot.RGBG_stacked(all_interpolated, extent=(lambdarange[0], lambdarange[-1], raw.ymax, raw.ymin), show_axes=True, xlabel="$\lambda$ (nm)", aspect=0.5 * len(lambdarange) / len(raw.x), saveto="TL_cutout_corrected.png", boost=boost)
plot.RGBG_stacked_with_graph(all_interpolated, x=lambdarange, extent=(lambdarange[0], lambdarange[-1], raw.ymax, raw.ymin), xlabel="$\lambda$ (nm)", aspect=0.5 * len(lambdarange) / len(raw.x), saveto="TL_cutout_corrected_spectrum.png", boost=boost)

plot.RGBG(RGBG, vmax=800, saveto="TL_split.png", size=30)
plot.RGBG(all_interpolated, vmax=800, saveto="TL_split_interpolated.png", size=30)

stacked = wavelength.stack(lambdarange, all_interpolated)
plot.plot_spectrum(stacked[:,0], stacked[:,1:])
