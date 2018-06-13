import numpy as np
import rawpy
from sys import argv
from matplotlib import pyplot as plt, patheffects as pe
from ispex.general import cut
from ispex.gamma import cos4f, find_I0
from ispex import raw, plot, io, wavelength
from scipy.optimize import curve_fit
from glob import glob


x = glob("results/bias/bias_stds_iso*.npy")
color_pattern = io.load_dng_raw("test_files/bias/0806a/IMG_0390.dng").raw_colors

isos = np.zeros(len(x))
mean_std = np.zeros((len(x), 4))
std_std = mean_std.copy()
for i,file in enumerate(x):
    iso = file.split(".")[0].split("_")[-1][3:]
    print(iso, end=", ")
    std = np.load(file)
    isos[i] = iso
    RGBG, _ = raw.pull_apart(std, color_pattern)
    mean_std[i] = [RGBG[...,0].mean(), RGBG[...,1::2].mean(), RGBG[...,2].mean(), std.mean()]
    std_std[i] = [RGBG[...,0].std(), RGBG[...,1::2].std(), RGBG[...,2].std(), std.std()]

for xmax, label in zip([1850, 300], ["", "_zoom"]):
    fig, axs = plt.subplots(2,2, sharex=True, sharey=True, figsize=(15,15), tight_layout=True)
    for j, colour, ax in zip(range(4), "RGBk", axs.ravel()):
        ax.errorbar(isos, mean_std[:,j], yerr=std_std[:,j], fmt='o', c=colour)
        ax.grid(ls="--")
    axs[1,0].set_xlabel("ISO") ; axs[1,1].set_xlabel("ISO")
    axs[0,0].set_ylabel("Mean RON") ; axs[1,0].set_ylabel("Mean RON")
    axs[1,0].set_xlim(0, xmax)
    fig.suptitle("Mean RON as function of ISO; error bars are std of RON")
    fig.savefig(f"results/bias/ISO_RON{label}.png")
    plt.close()