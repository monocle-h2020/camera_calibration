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
    gauss = gaussMd(std, sigma=10)
    std_RGBG, _= raw.pull_apart(std, colours)
    gauss_RGBG = gaussMd(std_RGBG, sigma=(0,5,5))
    vmin, vmax = gauss_RGBG.min(), gauss_RGBG.max()

    plot.hist_bias_ron_kRGB(std_RGBG, xlim=(0, 25), xlabel="Read noise (ADU)", saveto=results_readnoise/f"ADU_histogram_iso{iso}.pdf")

    plot.show_image(gauss, colorbar_label="Read noise (ADU)", saveto=results_readnoise/f"ADU_gauss_iso{iso}.pdf")
    for j, c in enumerate("RGBG"):
        X = "2" if j == 3 else ""
        plot.show_image(gauss_RGBG[j], colorbar_label="Read noise (ADU)", saveto=results_readnoise/f"ADU_{c}{X}_gauss_iso{iso}.pdf", colour=c, vmin=vmin, vmax=vmax)

    plot.show_RGBG(gauss_RGBG, colorbar_label=35*" "+"Read noise (ADU)", saveto=results_readnoise/f"ADU_all_gauss_iso{iso}.pdf", vmin=vmin, vmax=vmax)

    print(f"ISO: {iso} ; Mean: {std_RGBG.mean():.3f} ; Median: {np.median(std_RGBG):.3f} ; Max: {std_RGBG.max():.3f} ; Min: {std_RGBG.min():.3f} ; Standard deviation: {std_RGBG.std():.3f}")
