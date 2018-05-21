from matplotlib import pyplot as plt
import numpy as np
from sys import argv

from ispex import general, io, plot, wavelength_raw as wavelength, raw

filename = argv[1]

img = io.load_dng_raw(filename)
image_cut  = raw.cut_out_spectrum(img.raw_image)
colors_cut = raw.cut_out_spectrum(img.raw_colors)

RGBG, offsets = raw.pull_apart(image_cut, colors_cut)
plot.RGBG_stacked(RGBG, extent=(raw.xmin, raw.xmax, raw.ymax, raw.ymin), show_axes=True)

RGB = raw.to_RGB_array(image_cut, colors_cut)
RGB = plot._to_8_bit(RGB)
plot.Bayer(RGB, saveto="RGBG_Bayer.png")
del RGB  # conserve memory

def find_fluorescent_lines(RGBG, offsets):
    maxes = RGBG.argmax(axis=1)
    maxes = 2 * maxes + offsets[:,1]  # correct pixel offset from Bayer filter
    maxes += raw.xmin
    # line position per column in REAL image coordinates:
    lines = np.tile(np.nan, (3, raw.ymax - raw.ymin))
    lines[0, offsets[0,0]::2] = maxes[:,0]  # R
    lines[1, offsets[1,0]::2] = maxes[:,1]  # G
    lines[1, offsets[3,0]::2] = maxes[:,3]  # G2
    lines[2, offsets[2,0]::2] = maxes[:,2]  # B
    return lines

lines = find_fluorescent_lines(RGBG, offsets)

lines_fit = lines.copy()
for j in (0,1,2):  # fit separately for R, G, B
    idx = np.isfinite(lines[j])
    coeff = np.polyfit(raw.y[idx], lines[j][idx], wavelength.degree_of_spectral_line_fit)
    lines_fit[j] = np.polyval(coeff, raw.y)

plot.fluorescent_lines(raw.y, lines.T, lines_fit.T, saveto="line_locations.png")

wavelength_fits = wavelength.fit_many_wavelength_relations(RGBG_y, lines_fit.T)
coefficients, coefficients_fit = wavelength.fit_wavelength_coefficients(RGBG_y, wavelength_fits)

plot.wavelength_coefficients(RGBG_y, wavelength_fits, coefficients_fit)



raise Exception

def stack(x, RGBG, offsets, coeff, yoffset=0, lambdarange = np.arange(*wavelength_limits, 0.5)):
    wavelength_funcs = [wavelength_fit(c, *coeff) for c in range(yoffset, yoffset+rgb.shape[1])]
    wavelengths = np.array([f(x) for f in wavelength_funcs])
    # divide by nm/px to get intensity per nm
    rgb_new = rgb[:-1,:,:] / np.diff(wavelengths, axis=1).T[:,:,np.newaxis]
    interpolated = np.array([interpolate(wavelengths[i,:-1], rgb_new[:,i], lambdarange) for i in range(rgb.shape[1])])
    means = interpolated.mean(axis=0)
    return lambdarange, means

wavelengths, intensity_thick = wavelength.stack(x, thickF, coefficients, yoffset=y_thick[0])