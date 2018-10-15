import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import raw, plot, io
from phonecal.general import gaussMd

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
results_readnoise = results/"readnoise"

isos, stds  = io.load_stds  (folder, retrieve_value=io.split_iso)
colours     = io.load_colour(stacks)

low_iso = isos.argmin()
high_iso= isos.argmax()

for iso, std in zip(isos, stds):
    plot.hist_bias_ron(std, xlim=(0, 30), xlabel="Read noise (ADU)", saveto=results_readnoise/f"RON_hist_iso{iso}_ADU.png")

    std_RGBG, _= raw.pull_apart(std, colours)
    gauss_RGBG = gaussMd(std_RGBG, sigma=(0,5,5))
    vmin, vmax = gauss_RGBG.min(), gauss_RGBG.max()
    
    plot.hist_bias_ron_colour(std_RGBG, xlim=(0, 25), xlabel="Read noise (ADU)", saveto=results_readnoise/f"RON_hist_iso{iso}_ADU_colour.png")

    gauss = gaussMd(std, sigma=10)

    plot.show_image(gauss, colorbar_label="Read noise (ADU)", saveto=results_readnoise/f"RON_gauss_ADU_iso{iso}.png")

    for j, c in enumerate("RGBG"):
        X = "2" if j == 3 else ""
        plot.show_image(gauss_RGBG[j], colorbar_label="Read noise (ADU)", saveto=results_readnoise/f"RON_gauss_iso{iso}_{c}{X}_ADU.png", colour=c, vmin=vmin, vmax=vmax)
