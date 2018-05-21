from matplotlib import pyplot as plt
import numpy as np
from sys import argv

from ispex import general, io, plot, wavelength_raw as wavelength, raw

filename = argv[1]

img = io.load_dng_raw(filename)
imgarray = img.raw_image.astype(np.int16)
RGBG, offsets = raw.pull_apart(imgarray, img.raw_pattern)

cutout = raw.cut_out_spectrum(RGBG)

plot.RGBG_stacked(cutout, extent=(raw.range_x[0]*2, raw.range_x[1]*2, raw.range_y[1]*2, raw.range_y[0]*2), show_axes=True)

RGBG_y = np.arange(2*raw.range_y[0], 2*raw.range_y[1])

RGB = raw.to_RGB_array(img.raw_image, img.raw_colors)
RGB = plot._to_8_bit(RGB)
plot.Bayer(RGB, saveto="RGBG_Bayer.png")
del RGB  # conserve memory

def find_fluorescent_lines(RGBG, offsets):
    maxes = RGBG.argmax(axis=1)
    maxes += offsets[:,1]  # correct pixel offset from Bayer filter
    maxes += 0
    # line position per column in REAL image coordinates:
    lines = np.tile(np.nan, (3, 2*np.diff(raw.range_y)[0]))
    lines[0, offsets[0,0]::2] = maxes[:,0]  # R
    lines[1, offsets[1,0]::2] = maxes[:,1]  # G
    lines[1, offsets[3,0]::2] = maxes[:,3]  # G2
    lines[2, offsets[2,0]::2] = maxes[:,2]  # B
    return lines

lines = find_fluorescent_lines(cutout, offsets)
plot._rgbplot(RGBG_y, lines.T, func=plt.scatter)
plt.show()
plt.close()