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

lines, lines_fit = wavelength.find_fluorescent_lines(thickF, thinF)

plot.fluorescent_lines(y, lines, lines_fit)

wavelength_fits = wavelength.fit_many_wavelength_relations(y, lines_fit)
coefficients, coefficients_fit = wavelength.fit_wavelength_coefficients(y, wavelength_fits)

plot.wavelength_coefficients(y, wavelength_fits, coefficients_fit)

wavelengths, intensity_thick = wavelength.stack(x, thickF, coefficients, yoffset=y_thick[0])
wavelengths, intensity_thin  = wavelength.stack(x, thinF , coefficients, yoffset=y_thin[0] )

plot.plot_spectrum(wavelengths, intensity_thick, title="Stacked thick RGB spectrum", ylabel="C (au)", ylim=(0, None), xlim=wavelength.wavelength_limits)
plot.plot_spectrum(wavelengths, intensity_thin , title="Stacked thin RGB spectrum" , ylabel="C (au)", ylim=(0, None), xlim=wavelength.wavelength_limits)

for profile in [*intensity_thick.T, *intensity_thin.T]:
    res = wavelength.resolution(wavelengths, profile)
    print(f"Resolution: {res:.1f} nm")

wb = general.find_white_balance(data)
plot.plot_spectrum(wavelengths, intensity_thick/wb, title="Stacked thick RGB spectrum (post-WB)", ylabel="C (au)", ylim=(0, None), xlim=wavelength.wavelength_limits)
plot.plot_spectrum(wavelengths, intensity_thin/wb , title="Stacked thin RGB spectrum (post-WB)" , ylabel="C (au)", ylim=(0, None), xlim=wavelength.wavelength_limits)
