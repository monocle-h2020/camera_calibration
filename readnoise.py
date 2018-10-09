import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import raw, plot, io
from phonecal.general import gaussMd

folder = argv[1]
isos, means = io.load_means (folder, retrieve_value=io.split_iso, file=True)
isos, stds  = io.load_stds  (folder, retrieve_value=io.split_iso, file=True)
colours     = io.load_colour(folder)

gain_table = np.load(folder.replace("stacks", "products").replace("bias", "gain").strip("/")+"_lookup_table.npy")

low_iso = isos.argmin()
high_iso= isos.argmax()

for ind in (low_iso, high_iso):
    iso  = isos [ind]
    std  = stds [ind].copy()

    plt.figure(figsize=(10,7), tight_layout=True)
    plt.hist(std.ravel(), bins=np.linspace(0, 30, 500), color='k')
    plt.xlabel("Read noise (ADU)")
    plt.xlim(0, 30)
    plt.yscale("log")
    plt.ylabel("Frequency")
    plt.ylim(0.9, std.size)
    plt.grid(ls="--")
    plt.savefig(f"results/bias/RON_hist_iso{iso}_ADU.png")
    plt.close()

    gain = gain_table[1, iso]
    std  *= gain

    plt.figure(figsize=(10,7), tight_layout=True)
    plt.hist(std.ravel(), bins=np.linspace(0, 30, 500), color='k')
    plt.xlabel("Read noise (e$^-$)")
    plt.xlim(0, 30)
    plt.yscale("log")
    plt.ylabel("Frequency")
    plt.ylim(0.9, std.size)
    plt.grid(ls="--")
    plt.savefig(f"results/bias/RON_hist_iso{iso}_e.png")
    plt.close()

    std_RGBG, _ = raw.pull_apart(std, colours)

    fig, axs = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(5,5), squeeze=True, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0})
    std_RGB = [s.ravel() for s in std_RGBG[:3]]
    std_RGB[1] = np.concatenate((std_RGBG[1].ravel(), std_RGBG[3].ravel()))
    for s, c, ax in zip(std_RGB, "RGBG", axs):
        ax.hist(s.ravel(), bins=np.linspace(0, 15, 150), color=c, density=True)
        ax.grid(True)
    for ax in axs[:2]:
        ax.xaxis.set_ticks_position("none")
    axs[0].set_xlim(0, 15)
    axs[2].set_xlabel("Read noise (e$^-$)")
    axs[0].set_yscale("log")
    axs[1].set_ylabel("Probability density")
    fig.savefig(f"results/bias/RON_hist_iso{iso}_e_colour.png")
    plt.close()

    plot.imshow_gauss(std, sigma=10, colorbar_label="Read noise (e$^-$)", saveto=f"results/bias/RON_gauss_iso{iso}.png")

    for j, c in enumerate("RGBG"):
        X = "2" if j == 3 else ""
        plot.imshow_gauss(std_RGBG[j], sigma=5, colorbar_label="Read noise (e$^-$)", saveto=f"results/bias/RON_gauss_iso{iso}_{c}{X}.png", colour=c)
