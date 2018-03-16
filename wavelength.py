from matplotlib import pyplot as plt
import numpy as np
from sys import argv

from ispex import general, io, plot, wavelength

filename = argv[1]

col0 = 1530
col1 = 1911
col2 = 1970
col3 = 2315

row0 = 2150
row1 = 3900
x = np.arange(row0, row1)

data = io.load_dng(filename)
exif = io.load_exif(filename)

thick = data[row0:row1, col0:col1]
thin  = data[row0:row1, col2:col3]

plot.plot_photo(thick, extent=(col0, col1, row1, row0))
plot.plot_photo(thin , extent=(col2, col3, row1, row0))

thickF = general.gauss_filter(thick)
thinF  = general.gauss_filter(thin )

plot.plot_photo(thickF, extent=(col0, col1, row1, row0))
plot.plot_photo(thinF , extent=(col2, col3, row1, row0))

y_example = 100
plot.plot_spectrum(x, thick[:, y_example], xlabel="$x$ (px)", title=f"RGB values at $y = {y_example+col0}$", xlim=(row0,row1), ylim=(0,255))
plot.plot_spectrum(x, thickF[:, y_example], xlabel="$x$ (px)", title=f"RGB values at $y = {y_example+col0}$", xlim=(row0,row1), ylim=(0,255))
plot.plot_spectrum(x, thin[:, y_example], xlabel="$x$ (px)", title=f"RGB values at $y = {y_example+col2}$", xlim=(row0,row1), ylim=(0,255))
plot.plot_spectrum(x, thinF[:, y_example], xlabel="$x$ (px)", title=f"RGB values at $y = {y_example+col2}$", xlim=(row0,row1), ylim=(0,255))

y, lines, lines_fit = wavelength.find_fluorescent_lines(thickF, thinF, columns=(col0,col1,col2,col3), offset=row0)

plot.fluorescent_lines(y, lines, lines_fit)

wavelength_fits = wavelength.fit_many_wavelength_relations(y, lines_fit)
coefficients, coefficients_fit = wavelength.fit_wavelength_coefficients(y, wavelength_fits)

plot.wavelength_coefficients(y, wavelength_fits, coefficients_fit)

wavelengths, intensity_thick = wavelength.stack(x, thickF, coefficients, yoffset=col0)
wavelengths, intensity_thin  = wavelength.stack(x, thinF , coefficients, yoffset=col2)

plot.plot_spectrum(wavelengths, intensity_thick, title="Stacked thick RGB spectrum", ylabel="C (au)", ylim=(0, None), xlim=wavelength.wavelength_limits)
plot.plot_spectrum(wavelengths, intensity_thin , title="Stacked thin RGB spectrum" , ylabel="C (au)", ylim=(0, None), xlim=wavelength.wavelength_limits)

for profile in [*intensity_thick.T, *intensity_thin.T]:
    res = wavelength.resolution(wavelengths, profile)
    print(f"Resolution: {res:.1f} nm")

wb = general.find_white_balance(data, (row0, row1), col0)
plot.plot_spectrum(wavelengths, intensity_thick/wb, title="Stacked thick RGB spectrum (post-WB)", ylabel="C (au)", ylim=(0, None), xlim=wavelength.wavelength_limits)
plot.plot_spectrum(wavelengths, intensity_thin/wb , title="Stacked thin RGB spectrum (post-WB)" , ylabel="C (au)", ylim=(0, None), xlim=wavelength.wavelength_limits)
