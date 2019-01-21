from matplotlib import pyplot as plt
import numpy as np
from sys import argv
from phonecal import general, io, plot, wavelength, raw

file = io.path_from_input(argv)

img = io.load_dng_raw(file)
image_cut  = img.raw_image [760:1470, 2150:3900]
colors_cut = img.raw_colors[760:1470, 2150:3900]

RGBG,_ = raw.pull_apart(image_cut, colors_cut)
plot.show_RGBG(RGBG)

RGBG_gauss = general.gaussMd(RGBG, sigma=(0,0,5))
plot.show_RGBG(RGBG_gauss)

raise Exception

lines = wavelength.find_fluorescent_lines(RGBG_gauss, offsets)

lines_fit = wavelength.fit_fluorescent_lines(lines)

plot.fluorescent_lines(raw.y, lines.T, lines_fit.T, saveto="TL_lines.png")

wavelength_fits = wavelength.fit_many_wavelength_relations(raw.y, lines_fit.T)
coefficients, coefficients_fit = wavelength.fit_wavelength_coefficients(raw.y, wavelength_fits)
plot.wavelength_coefficients(raw.y, wavelength_fits, coefficients_fit, saveto="TL_coeff.png")

wavelength.save_coefficients(coefficients, saveto="wavelength_solution.npy")

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
