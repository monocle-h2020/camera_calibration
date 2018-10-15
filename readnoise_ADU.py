import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import raw, plot, io
from phonecal.general import gaussMd

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
isos, stds  = io.load_stds  (folder, retrieve_value=io.split_iso)
colours     = io.load_colour(stacks)

low_iso = isos.argmin()
high_iso= isos.argmax()

for iso, std in zip(isos, stds):
    plt.figure(figsize=(10,7), tight_layout=True)
    plt.hist(std.ravel(), bins=np.linspace(0, 30, 500), color='k')
    plt.xlabel("Read noise (ADU)")
    plt.xlim(0, 30)
    plt.yscale("log")
    plt.ylabel("Frequency")
    plt.ylim(0.9, std.size)
    plt.grid(ls="--")
    plt.savefig(results/f"bias/RON_hist_iso{iso}_ADU.png")
    plt.close()

    std_RGBG, _= raw.pull_apart(std, colours)
    gauss_RGBG = gaussMd(std_RGBG, sigma=(0,5,5))
    vmin, vmax = gauss_RGBG.min(), gauss_RGBG.max()

    fig, axs = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(5,5), squeeze=True, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0})
    std_RGB = [s.ravel() for s in std_RGBG[:3]]
    std_RGB[1] = np.concatenate((std_RGBG[1].ravel(), std_RGBG[3].ravel()))
    for s, c, ax in zip(std_RGB, "RGBG", axs):
        ax.hist(s.ravel(), bins=np.linspace(0, 15, 150), color=c, density=True)
        ax.grid(True)
    for ax in axs[:2]:
        ax.xaxis.set_ticks_position("none")
    axs[0].set_xlim(0, 15)
    axs[2].set_xlabel("Read noise (ADU)")
    axs[0].set_yscale("log")
    axs[1].set_ylabel("Probability density")
    fig.savefig(results/f"bias/RON_hist_iso{iso}_ADU_colour.png")
    plt.close()

    gauss = gaussMd(std, sigma=10)

    plot.show_image(gauss, colorbar_label="Read noise (ADU)", saveto=results/f"bias/RON_gauss_ADU_iso{iso}.png")

    for j, c in enumerate("RGBG"):
        X = "2" if j == 3 else ""
        plot.show_image(gauss_RGBG[j], colorbar_label="Read noise (ADU)", saveto=results/f"bias/RON_gauss_iso{iso}_{c}{X}_ADU.png", colour=c, vmin=vmin, vmax=vmax)
