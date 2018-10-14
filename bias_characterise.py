import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import raw, plot, io
from phonecal.general import gaussMd

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")

isos, means = io.load_means (folder, retrieve_value=io.split_iso)
colours     = io.load_colour(stacks)

savefolder = results/"bias"

xmax = phone["software"]["bias"] + 25
xmin = max(phone["software"]["bias"] - 25, 0)

for iso, mean in zip(isos, means):
    plt.figure(figsize=(10,7), tight_layout=True)
    plt.hist(mean.ravel(), bins=np.linspace(xmin, xmax, 250), color='k')
    plt.xlabel("Mean bias (ADU)")
    plt.xlim(xmin, xmax)
    plt.ylim(0.9, mean.size)
    plt.yscale("log")
    plt.ylabel("Frequency")
    plt.ylim(ymin=0.9)
    plt.grid(ls="--")
    plt.savefig(savefolder/f"Bias_mean_hist_iso{iso}.png")
    plt.close()

    gauss = gaussMd(mean, sigma=10)

    plot.show_image(gauss, colorbar_label="Mean bias (ADU)", saveto=savefolder/f"Bias_mean_gauss_iso{iso}.png")

    mean_RGBG, _ = raw.pull_apart(mean, colours)
    gauss_RGBG = gaussMd(mean_RGBG, sigma=(0,5,5))
    vmin, vmax = gauss_RGBG.min(), gauss_RGBG.max()

    for j, c in enumerate("RGBG"):
        X = "2" if j == 3 else ""
        plot.show_image(gauss_RGBG[j], colorbar_label="Mean bias (ADU)", saveto=savefolder/f"Bias_mean_gauss_iso{iso}_{c}{X}.png", colour=c, vmin=vmin, vmax=vmax)
    print(iso)
