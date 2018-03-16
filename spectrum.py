from matplotlib import pyplot as plt
import numpy as np
from sys import argv

from ispex import general, io, plot, wavelength
from ispex.general import x, y, y_thin, y_thick, x_spectrum

filename = argv[1]

data = io.load_dng(filename)
exif = io.load_exif(filename)

thick, thin = general.split_spectrum(data)

plot.plot_photo(thick, extent=(*y_thick, *x_spectrum[::-1]))
plot.plot_photo(thin , extent=(*y_thin , *x_spectrum[::-1]))

thickF = general.gauss_filter(thick)
thinF  = general.gauss_filter(thin )

plot.plot_photo(thickF, extent=(*y_thick, *x_spectrum[::-1]))
plot.plot_photo(thinF , extent=(*y_thin , *x_spectrum[::-1]))

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