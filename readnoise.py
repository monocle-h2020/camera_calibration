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

    std_gauss = gaussMd(std, sigma=10)

    plt.figure(figsize=(20,10), tight_layout=True)
    img = plt.imshow(std_gauss)
    plot.colorbar(img)
    plt.savefig(f"results/bias/RON_gauss_iso{iso}.png")
    plt.close()

    std_RGBG, _ = raw.pull_apart(std, colours)
    for j, c in enumerate("RGBG"):
        X = "2" if j == 3 else ""
        std_gauss_C = gaussMd(std_RGBG[j], sigma=5)
        plt.figure(figsize=(20,10), tight_layout=True)
        img = plt.imshow(std_gauss_C, cmap=plot.cmaps[c+"r"])
        plot.colorbar(img)
        plt.savefig(f"results/bias/RON_gauss_iso{iso}_{c}{X}.png")
        plt.close()
