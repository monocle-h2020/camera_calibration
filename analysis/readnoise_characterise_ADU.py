from sys import argv
from matplotlib import pyplot as plt
from spectacle import raw, plot, io, analyse
from spectacle.general import gaussMd

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
results_readnoise = results/"readnoise"

isos, stds  = io.load_stds  (folder, retrieve_value=io.split_iso)
colours     = io.load_colour(stacks)

table = analyse.statistics(stds, prefix_column=isos, prefix_column_header="ISO")
print(table)

for iso, std in zip(isos, stds):
    gauss = gaussMd(std, sigma=10)
    std_RGBG, _= raw.pull_apart(std, colours)
    gauss_RGBG = gaussMd(std_RGBG, sigma=(0,5,5))
    vmin, vmax = gauss_RGBG.min(), gauss_RGBG.max()

    plot.histogram_RGB(std_RGBG, xlim=(0, 25), xlabel="Read noise (ADU)", saveto=results_readnoise/f"ADU_histogram_iso{iso}.pdf")

    plot.show_image(gauss, colorbar_label="Read noise (ADU)", saveto=results_readnoise/f"ADU_gauss_iso{iso}.pdf")
    plot.show_image_RGBG2(gauss_RGBG, colorbar_label="Read noise (ADU)", saveto=results_readnoise/f"ADU_gauss_iso{iso}.pdf", vmin=vmin, vmax=vmax)
