import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import raw, plot, io, iso, analyse
from spectacle.general import gaussMd

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
results_readnoise = results/"readnoise"

isos, stds  = io.load_stds  (folder, retrieve_value=io.split_iso)
colours     = io.load_colour(stacks)

lookup_table = io.read_iso_lookup_table(products)

stds_normalised = np.array([iso.normalise(std_, ISO, lookup_table) for std_, ISO in zip(stds, isos)])

table = analyse.statistics(stds_normalised, prefix_column=isos, prefix_column_header="ISO")
print(table)

for ISO, std in zip(isos, stds_normalised):
    gauss = gaussMd(std, sigma=10)
    std_RGBG, _= raw.pull_apart(std, colours)
    gauss_RGBG = gaussMd(std_RGBG, sigma=(0,5,5))
    vmin, vmax = gauss_RGBG.min(), gauss_RGBG.max()

    plot.hist_bias_ron_kRGB(std_RGBG, xlim=(0, 15), xlabel="Read noise (norm. ADU)", saveto=results_readnoise/f"normalised_histogram_iso{ISO}.pdf")

    plot.show_image(gauss, colorbar_label="Read noise (norm. ADU)", saveto=results_readnoise/f"normalised_gauss_iso{ISO}.pdf")
    for j, c in enumerate("RGBG"):
        X = "2" if j == 3 else ""
        plot.show_image(gauss_RGBG[j], colorbar_label="Read noise (norm. ADU)", saveto=results_readnoise/f"normalised_{c}{X}_gauss_iso{ISO}.pdf", colour=c, vmin=vmin, vmax=vmax)

    plot.show_RGBG(gauss_RGBG, colorbar_label=25*" "+"Read noise (norm. ADU)", saveto=results_readnoise/f"normalised_all_gauss_iso{ISO}.pdf", vmin=vmin, vmax=vmax)
