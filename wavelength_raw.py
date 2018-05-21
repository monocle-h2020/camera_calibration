from matplotlib import pyplot as plt
import numpy as np
from sys import argv

from ispex import general, io, plot, wavelength_raw as wavelength, raw

filename = argv[1]

boost=2.5

img = io.load_dng_raw(filename)
image_cut  = raw.cut_out_spectrum(img.raw_image)
colors_cut = raw.cut_out_spectrum(img.raw_colors)

RGBG, offsets = raw.pull_apart(image_cut, colors_cut)
plot.RGBG_stacked(RGBG, extent=(raw.xmin, raw.xmax, raw.ymax, raw.ymin), show_axes=True, saveto="TL_cutout.png", boost=boost)

RGB = raw.to_RGB_array(image_cut, colors_cut)
RGB = plot._to_8_bit(RGB, boost=boost)
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

plot.fluorescent_lines(raw.y, lines.T, lines_fit.T, saveto="TL_lines.png")

wavelength_fits = wavelength.fit_many_wavelength_relations(raw.y, lines_fit.T)
coefficients, coefficients_fit = wavelength.fit_wavelength_coefficients(raw.y, wavelength_fits)

plot.wavelength_coefficients(raw.y, wavelength_fits, coefficients_fit)

wavelengths_cut = np.array([np.polyval(c, raw.x) for c in coefficients_fit])
wavelengths_split, offsets = raw.pull_apart(wavelengths_cut, colors_cut)
#nm_diff = np.diff(wavelengths_split, axis=1)/2
#nm_width = nm_diff[:,1:,:] + nm_diff[:,:-1,:]
#wavelengths_split = wavelengths_split[:,1:-1,:]
#RGBG = RGBG[:,1:-1,:] / nm_width
lambdarange = np.arange(340, 760, 0.5)

def interpolate(wavelength_array, color_value_array, lambdarange):
    interpolated = np.array([np.interp(lambdarange, wavelengths, color_values)
    for wavelengths, color_values in zip(wavelength_array, color_value_array)])
    return interpolated

all_interpolated = np.array([interpolate(wavelengths_split[:,:,c], RGBG[:,:,c], lambdarange) for c in range(4)])
all_interpolated = all_interpolated.T.swapaxes(0,1)
plot.RGBG_stacked(all_interpolated, extent=(lambdarange[0], lambdarange[-1], raw.ymax, raw.ymin), show_axes=True, xlabel="$\lambda$ (nm)", aspect=0.5 * len(lambdarange) / len(raw.x), saveto="TL_cutout_corrected.png", boost=boost)
plot.RGBG_stacked_with_graph(all_interpolated, x=lambdarange, extent=(lambdarange[0], lambdarange[-1], raw.ymax, raw.ymin), xlabel="$\lambda$ (nm)", aspect=0.5 * len(lambdarange) / len(raw.x), saveto="TL_cutout_corrected_spectrum.png", boost=boost)