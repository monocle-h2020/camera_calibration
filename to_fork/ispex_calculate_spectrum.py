import numpy as np
from sys import argv
from spectacle import raw, plot, io, wavelength, calibrate

file = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(file)

img  = io.load_raw_file(file)
exif = io.load_exif(file)

values = img.raw_image.astype(np.float32)
values = calibrate.correct_bias(root, values)

image_cut  = values        [760:1000, 2150:3900]
colors_cut = img.raw_colors[760:1000, 2150:3900]
x = np.arange(2150, 3900)
y = np.arange(760 , 1000)

RGBG = raw.demosaick(colors_cut, image_cut)
plot.show_RGBG(RGBG)

coefficients = wavelength.load_coefficients(results/"ispex/wavelength_solution.npy")
wavelengths_cut = wavelength.calculate_wavelengths(coefficients, x, y)
wavelengths_split = raw.demosaick(colors_cut, wavelengths_cut)

lambdarange, all_interpolated = wavelength.interpolate_multi(wavelengths_split, RGBG)
stacked = wavelength.stack(lambdarange, all_interpolated)
plot.plot_spectrum(stacked[0], stacked[1:], saveto=results/"ispex"/(file.stem+".pdf"))

np.save(results/"ispex"/(file.stem+".npy"), stacked)
