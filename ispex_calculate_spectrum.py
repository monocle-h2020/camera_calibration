from matplotlib import pyplot as plt
import numpy as np
from sys import argv
from phonecal import general, io, plot, wavelength, raw
from phonecal.general import gaussMd

file = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(file)

img = io.load_dng_raw(file)
exif = io.load_exif(file)
image_cut  = img.raw_image [760:1000, 2150:3900]
colors_cut = img.raw_colors[760:1000, 2150:3900]
x = np.arange(2150, 3900)
y = np.arange(760 , 1000)

RGBG,_ = raw.pull_apart(image_cut, colors_cut)
plot.show_RGBG(RGBG)
print("Split spectrum into RGBG")

RGBG_gauss = gaussMd(RGBG, sigma=(0,5,0))
plot.show_RGBG(RGBG_gauss)
print("Gaussed spectrum")

raise Exception

y_example = 100
plot.plot_spectrum(general.x, thick[:, y_example], xlabel="$x$ (px)", title=f"RGB values at $y = {y_example+y_thick[0]}$", xlim=x_spectrum, ylim=(0,255))
plot.plot_spectrum(general.x, thickF[:, y_example], xlabel="$x$ (px)", title=f"RGB values at $y = {y_example+y_thick[0]}$", xlim=x_spectrum, ylim=(0,255))
plot.plot_spectrum(general.x, thin[:, y_example], xlabel="$x$ (px)", title=f"RGB values at $y = {y_example+y_thin[0]}$", xlim=x_spectrum, ylim=(0,255))
plot.plot_spectrum(general.x, thinF[:, y_example], xlabel="$x$ (px)", title=f"RGB values at $y = {y_example+y_thin[0]}$", xlim=x_spectrum, ylim=(0,255))

coefficients = wavelength.load_coefficients("wavelength.npy")

wavelengths, intensity_thick = wavelength.stack(x, thickF, coefficients, yoffset=y_thick[0])
wavelengths, intensity_thin  = wavelength.stack(x, thinF , coefficients, yoffset=y_thin[0] )

plot.plot_spectrum(wavelengths, intensity_thick, title="Stacked thick RGB spectrum", ylabel="C (au)", ylim=(0, None), xlim=wavelength.wavelength_limits)
plot.plot_spectrum(wavelengths, intensity_thin , title="Stacked thin RGB spectrum" , ylabel="C (au)", ylim=(0, None), xlim=wavelength.wavelength_limits)

wb = general.find_white_balance(data)
intensity_thick /= wb
intensity_thin  /= wb
plot.plot_spectrum(wavelengths, intensity_thick, title="Stacked thick RGB spectrum (post-WB)", ylabel="C (au)", ylim=(0, None), xlim=wavelength.wavelength_limits)
plot.plot_spectrum(wavelengths, intensity_thin , title="Stacked thin RGB spectrum (post-WB)" , ylabel="C (au)", ylim=(0, None), xlim=wavelength.wavelength_limits)

filter_curves = np.load("filtercurves.npy")
intensity_thick /= filter_curves
intensity_thin  /= filter_curves
plot.plot_spectrum(wavelengths, intensity_thick, title="Stacked thick RGB spectrum (post-FC)", ylabel="C (au)", ylim=(0, None), xlim=wavelength.wavelength_limits)
plot.plot_spectrum(wavelengths, intensity_thin , title="Stacked thin RGB spectrum (post-FC)" , ylabel="C (au)", ylim=(0, None), xlim=wavelength.wavelength_limits)
