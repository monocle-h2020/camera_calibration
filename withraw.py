import numpy as np
import rawpy
from sys import argv
from matplotlib import pyplot as plt, patheffects as pe
from ispex.general import gauss_filter
from ispex import raw
from ispex import plot, io

filename = argv[1]
handle = filename.split()

img = io.load_dng_raw(filename)
imgarray = img.raw_image.astype(np.int16)
RGBG, offsets = raw.pull_apart(imgarray, img.raw_pattern)

plot.RGBG_stacked(RGBG, saveto="RGBG_stacked.png")

cutout = raw.cut_out_spectrum(RGBG)
plot.RGBG_stacked(cutout, saveto="RGBG_stacked_cutout.png", boost=5)

plot.plot_spectrum(raw.x, cutout.mean(0), xlabel="Pixel", ylabel="Mean RGBG value")

plot.RGBG_stacked_with_graph(cutout, saveto="RGBG_spectrum.png")

plot.RGBG(RGBG, vmax=800, saveto="RGBG_split.png", size=30)
RGBGg = gauss_filter(RGBG)
Rg, Gg, Bg, G2g = raw.split_RGBG(RGBGg)
plot.RGBG(RGBGg, vmax=800, saveto="RGBG_split_gauss.png", size=30)