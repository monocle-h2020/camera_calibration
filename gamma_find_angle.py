import numpy as np
import rawpy
from sys import argv
from matplotlib import pyplot as plt, patheffects as pe
from ispex.general import gauss_filter
from ispex import raw, plot, io, wavelength
from scipy.optimize import curve_fit
from glob import glob

folder = argv[1]
files = glob(folder+"/*.dng")

def cut(arr, x=250, y=250):
    return arr[y:-y, x:-x]

def cos4(d, p, a):
    return a*(np.cos(d/p))**4

def cos4f(d, f, a):
    return a*(np.cos(np.arctan(d/f)))**4

def find_I0(rgbg, distances, radius=100):
    return rgbg[distances < radius].mean()

p0s = []
v0s = []

for filename in files:
    img = io.load_dng_raw(filename)

    image_cut  = cut(img.raw_image)
    colors_cut = cut(img.raw_colors)

    RGBG, offsets = raw.pull_apart(image_cut, colors_cut)
    #plot.RGBG_stacked(RGBG, show_axes=True, boost=1)

    x = np.arange(image_cut.shape[1])
    y = np.arange(image_cut.shape[0])
    X, Y = np.meshgrid(x, y)
    (x0, y0) = (len(x) / 2, len(y) / 2)
    D = np.sqrt((X - x0)**2 + (Y - y0)**2)
    D_split, offsets = raw.pull_apart(D, colors_cut)

    RGBG_norm = RGBG.astype(np.float) - 528

    for j in range(4):
        popt, pcov = curve_fit(cos4f, D_split[..., j].ravel(), RGBG_norm[..., j].ravel(), p0=[3000, 1000])
        print(popt, np.sqrt(np.diag(pcov)))
        p0s.append(popt[0])
        v0s.append(pcov[0,0])

p0s = np.array(p0s)
v0s = np.array(v0s)

mean = np.average(p0s, weights=1/v0s)
np.save("pixel_angle.npy", mean)
