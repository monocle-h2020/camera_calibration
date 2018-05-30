import numpy as np
import rawpy
from sys import argv
from matplotlib import pyplot as plt, patheffects as pe
from ispex.general import gauss_filter
from ispex.gamma import I_range
from ispex import raw, plot, io, wavelength
from scipy.optimize import curve_fit
from glob import glob

folder = argv[1]

colours = "RGB"

bin_width = I_range[1] - I_range[0]
bin_centers = I_range[:-1] + bin_width/2
means = np.zeros((3, len(bin_centers)))
stds = means.copy()
nrs = means.copy().astype(np.uint32)

for i, b in enumerate(I_range[:-1]):
    for j, c in enumerate(colours):
        filename = f"{folder}/{c}_{b:.3f}.npy"
        data = np.load(filename)
        nrs[j,i] = len(data)
        means[j,i] = data.mean()
        stds[j,i] = data.std()

for j, c in enumerate(colours):
    idx = np.where((nrs[j] > 5000) & (means[j] < 4000))
    plt.figure(figsize=(15,8))
    plt.scatter(bin_centers[idx], means[j][idx], c=c)
    plt.ylim(0, 4500)
    plt.xlim(0, I_range[np.nanargmax(means[j][idx])])
    plt.xlabel("Intensity")
    plt.ylabel(f"Mean {c} value")
    plt.axhline(4096, c='k', ls="--")
    plt.savefig(f"{c}_linearity.png")
    plt.show()
    plt.close()

for j, c in enumerate(colours):
    idx = np.where((nrs[j] > 5000) & (means[j] < 4000))
    plt.figure(figsize=(15,8))
    plt.scatter(bin_centers[idx], stds[j][idx]**2, c=c)
    plt.xlim(0, I_range[np.nanargmax(means[j][idx])])
    plt.xlabel("Intensity")
    plt.ylabel(f"$\\sigma^2_{c}$")
    plt.savefig(f"{c}_sigma.png")
    plt.show()
    plt.close()

for j, c in enumerate(colours):
    idx = np.where((nrs[j] > 5000) & (means[j] < 4000))
    plt.figure(figsize=(15,8))
    plt.scatter(bin_centers[idx], stds[j][idx]**2 / (means[j][idx]-531), c=c)
    plt.xlim(0, I_range[np.nanargmax(means[j][idx])])
    plt.xlabel("Intensity")
    plt.ylabel(f"$\\sigma^2_{c} / (\\mu_{c} - 531)$")
    plt.savefig(f"{c}_sigma_over_mu.png")
    plt.show()
    plt.close()

for j, c in enumerate(colours):
    idx = np.where((nrs[j] > 5000) & (means[j] < 4000))
    p = np.polyfit(bin_centers[idx], means[j][idx], 1)
    plt.scatter(bin_centers[idx], means[j][idx], c=c)
    plt.plot(bin_centers, np.polyval(p, bin_centers), c='k')
    plt.ylim(0, 4500)
    plt.xlim(0, I_range[np.nanargmax(means[j])])
    plt.xlabel("Intensity")
    plt.ylabel(f"Mean {c} value")
    plt.axhline(4096, c='k', ls="--")
    plt.show()
    plt.close()