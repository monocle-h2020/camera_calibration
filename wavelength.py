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

thickF = general.gauss_filter(thick, sigma=9)
thinF  = general.gauss_filter(thin , sigma=9)

plot.plot_photo(thickF, extent=(*y_thick, *x_spectrum[::-1]))
plot.plot_photo(thinF , extent=(*y_thin , *x_spectrum[::-1]))

y_example = 100
plot.plot_spectrum(general.x, thick[:, y_example], xlabel="$x$ (px)", title=f"RGB values at $y = {y_example+y_thick[0]}$", xlim=x_spectrum, ylim=(0,255))
plot.plot_spectrum(general.x, thickF[:, y_example], xlabel="$x$ (px)", title=f"RGB values at $y = {y_example+y_thick[0]}$", xlim=x_spectrum, ylim=(0,255))
plot.plot_spectrum(general.x, thin[:, y_example], xlabel="$x$ (px)", title=f"RGB values at $y = {y_example+y_thin[0]}$", xlim=x_spectrum, ylim=(0,255))
plot.plot_spectrum(general.x, thinF[:, y_example], xlabel="$x$ (px)", title=f"RGB values at $y = {y_example+y_thin[0]}$", xlim=x_spectrum, ylim=(0,255))

lines, lines_fit = wavelength.find_fluorescent_lines(thickF, thinF)

plot.fluorescent_lines(y, lines, lines_fit)

wavelength_fits = wavelength.fit_many_wavelength_relations(y, lines_fit)
coefficients, coefficients_fit = wavelength.fit_wavelength_coefficients(y, wavelength_fits)

plot.wavelength_coefficients(y, wavelength_fits, coefficients_fit)

wavelength.save_coefficients(coefficients)
