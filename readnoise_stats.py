import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from ispex import raw, plot, io
from ispex.general import gaussMd
from glob import glob

def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

x = glob("results/bias/bias_stds_iso*.npy")
color_pattern = io.load_colors("test_files/bias/0806a/IMG_0390.dng")

isos = np.zeros(len(x))
mean_std = np.zeros((len(x), 4))
std_std = mean_std.copy()
for i,file in enumerate(x):
    iso = file.split(".")[0].split("_")[-1][3:]
    print(iso, end=", ")
    std = np.load(file)
    mean = np.load(file.replace("stds", "mean"))
    isos[i] = iso
    handle = f"_iso{iso}"

    RGBG, _ = raw.pull_apart(std, color_pattern)
    reshaped = [RGBG[...,0], RGBG[...,1::2], RGBG[...,2], std]

    fig, axs = plt.subplots(2,2, sharex=True, sharey=True, figsize=(15,15), tight_layout=True)
    for j, colour, ax, resh in zip(range(4), "RGBk", axs.ravel(), reshaped):
        ax.hist(resh.ravel(), bins=np.arange(0,25,0.2), color=colour)
        ax.grid(ls="--")
    axs[1,0].set_xlabel("$\sigma$") ; axs[1,1].set_xlabel("$\sigma$")
    axs[0,0].set_ylabel("$f$") ; axs[1,0].set_ylabel("$f$")
    axs[0,0].set_yscale("log") ; axs[0,0].set_ylim(ymin=1)
    fig.savefig(f"results/bias/Bias_std_hist{handle}.png")
    plt.close()

    plot.bitmap(mean, saveto=f"results/bias/Bias_mean{handle}.png")
    plot.bitmap(std , saveto=f"results/bias/Bias_std_im{handle}.png")

    G = gaussMd(std, sigma=10)
    plot.imshow_tight(G, saveto=f"results/bias/Bias_std_gauss{handle}.png")

    for j, c in enumerate("RGBG"):
        G = gaussMd(RGBG[...,j], sigma=10)
        X = "2" if j == 3 else ""
        plot.imshow_tight(G, saveto=f"results/bias/Bias_std_gauss{handle}_{c}{X}.png", cmap=plot.cmaps[c+"r"])

        plt.figure(figsize=(25, 20))
        img = plt.imshow(G, interpolation="none", aspect="equal", cmap=plot.cmaps[c+"r"])
        plt.axis("off")
        colorbar(img)
        plt.tight_layout()
        plt.savefig(f"results/bias/Bias_std_im{handle}_{c}{X}_cb.png")
        plt.close()