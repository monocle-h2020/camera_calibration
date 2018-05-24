import numpy as np
import rawpy
from sys import argv
from matplotlib import pyplot as plt, patheffects as pe
from ispex.general import gauss_filter
from ispex import raw, plot, io, wavelength
from scipy.optimize import curve_fit
from glob import glob

folder = argv[1]

colours = "RGB"

for colour in colours:
    files = glob(f"{folder}/{colour}*.npy")
    arr = np.array([np.load(f) for f in files])


for colour, C, D in zip(colours, RGB, DRGB):
    st = int(len(D)/nr_points)
    o = 0
    plt.hist(C, bins=np.arange(0, 5000, 100), color=colour)
    plt.xlim(0, 4096)
    plt.xlabel(colour+" value")
    plt.ylabel("# pixels")
    plt.show()
    plt.close()

    d_range = np.arange(0, D.max(), 10)

    I0 = find_I0(C-528, D)

    plt.scatter(D[o::st], C[o::st], c=colour, s=10, label="Data")
    plt.plot(d_range, cos4f(d_range, pixel_angle, I0)+528, c='k', label="$\cos^4$ fit")
    plt.xlabel("Distance from center (px)")
    plt.ylabel(colour+" value")
    plt.legend(loc="lower left")
    plt.show()
    plt.close()

    I = cos4(D, pixel_angle, malus(polarisation_angle))

    plt.scatter(I[o::st], C[o::st], c=colour, s=10)
    plt.xlabel("Intensity")
    plt.ylabel("RGB value")
    plt.xlim(0, 1)
    plt.ylim(0, 4096)
    plt.title(f"{colour} linearity ({len(I[o::st])/len(I)*100:.2f}% of data shown)")
    plt.show()
    plt.close()

    np.save(f"gamma/{colour}_{handle}.npy", np.dstack((I,C)))