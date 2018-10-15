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
    plot.hist_bias_ron(mean, xlim=(xmin, xmax), xlabel="Bias (ADU)", saveto=savefolder/f"Bias_mean_hist_iso{iso}.png")

    gauss = gaussMd(mean, sigma=10)

    plot.show_image(gauss, colorbar_label="Mean bias (ADU)", saveto=savefolder/f"Bias_mean_gauss_iso{iso}.png")

    mean_RGBG, _ = raw.pull_apart(mean, colours)
    gauss_RGBG = gaussMd(mean_RGBG, sigma=(0,5,5))
    vmin, vmax = gauss_RGBG.min(), gauss_RGBG.max()
    
    plot.hist_bias_ron_colour(mean_RGBG, xlim=(xmin, xmax), xlabel="Bias (ADU)", saveto=savefolder/f"Bias_mean_gauss_iso{iso}_colour.png")

    for j, c in enumerate("RGBG"):
        X = "2" if j == 3 else ""
        plot.show_image(gauss_RGBG[j], colorbar_label="Mean bias (ADU)", saveto=savefolder/f"Bias_mean_gauss_iso{iso}_{c}{X}.png", colour=c, vmin=vmin, vmax=vmax)
    print(iso)
