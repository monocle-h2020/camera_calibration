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
rms = np.zeros((len(x), 4))
std_std = rms.copy()
for i,file in enumerate(x):
    iso = file.split(".")[0].split("_")[-1][3:]
    print(iso, end=", ")
    std = np.load(file)
    isos[i] = iso
    RGBG, _ = raw.pull_apart(std, color_pattern)
    reshaped = [RGBG[...,0], RGBG[...,1::2], RGBG[...,2], std]
    rms[i] = [np.sqrt(np.mean(resh**2)) for resh in reshaped]
    std_std[i] = [np.std(resh) for resh in reshaped]

for xmax, label in zip([1850, 300], ["", "_zoom"]):
    fig, axs = plt.subplots(2,2, sharex=True, sharey=True, figsize=(15,15), tight_layout=True)
    for j, colour, ax in zip(range(4), "RGBk", axs.ravel()):
        ax.errorbar(isos, rms[:,j], yerr=std_std[:,j], fmt='o', c=colour)
        ax.grid(ls="--")
    axs[1,0].set_xlabel("ISO") ; axs[1,1].set_xlabel("ISO")
    axs[0,0].set_ylabel("RMS RON") ; axs[1,0].set_ylabel("RMS RON")
    axs[1,0].set_xlim(0, xmax)
    fig.suptitle("RMS RON as function of ISO; error bars are std of RON")
    fig.savefig(f"results/bias/ISO_RON{label}.png")
    plt.close()