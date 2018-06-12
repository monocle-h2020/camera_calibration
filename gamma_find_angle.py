import numpy as np
import rawpy
from sys import argv
from matplotlib import pyplot as plt, patheffects as pe
from ispex.general import cut
from ispex.gamma import cos4f, find_I0
from ispex import raw, plot, io, wavelength
from scipy.optimize import curve_fit
from glob import glob

folder = argv[1]
files = glob(folder+"/*.dng")

p0s = [[], [], []]
v0s = [[], [], []]

for filename in files:
    print(filename)
    img = io.load_dng_raw(filename)

    image_cut  = cut(img.raw_image)
    colors_cut = cut(img.raw_colors)

    RGBG, offsets = raw.pull_apart(image_cut, colors_cut)

    x = np.arange(image_cut.shape[1])
    y = np.arange(image_cut.shape[0])
    X, Y = np.meshgrid(x, y)
    (x0, y0) = (len(x) / 2, len(y) / 2)
    D = np.sqrt((X - x0)**2 + (Y - y0)**2)
    D_split, offsets = raw.pull_apart(D, colors_cut)

    RGBG_norm = RGBG.astype(np.float)

    for j in range(4):
        C = RGBG_norm[..., j]
        if len(np.where(C > 4000)[0]):
            continue
        D = D_split[..., j]
        I0 = find_I0(C, D)
        popt, pcov = curve_fit(cos4f, D.ravel(), C.ravel(), p0=[3000, 1000, 528])
        print(popt, np.sqrt(np.diag(pcov)))
        col = j if j <= 2 else 1
        p0s[col].append(popt[0])
        v0s[col].append(pcov[0,0])
    print("----")

means = np.empty(3)
errs  = means.copy()
for j, p, v in zip(range(3), p0s, v0s):
    p = np.array(p)
    v = np.array(v)
    sumw = np.sum(1/v)
    mean = np.sum(1/v * p) / sumw
    #print(mean)
    #print(p.astype(int))
    #print(v.astype(int))
    #print("---")
    means[j] = mean
    err = np.sum(1/v * (p-mean)**2) / (sumw - np.sum(1/v**2)/sumw)  # Bessel corrected
    errs[j] = np.sqrt(err)

#np.save("pixel_angle.npy", means)
