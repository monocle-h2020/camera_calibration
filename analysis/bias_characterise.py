from sys import argv
from matplotlib import pyplot as plt
from spectacle import raw, plot, io, analyse
from spectacle.general import gaussMd

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.load_metadata(root)

isos, means = io.load_means (folder, retrieve_value=io.split_iso)
colours     = io.load_colour(stacks)

table = analyse.statistics(means, prefix_column=isos, prefix_column_header="ISO")
print(table)

savefolder = results/"bias"

xmax = phone["software"]["bias"] + 25
xmin = max(phone["software"]["bias"] - 25, 0)

for iso, mean in zip(isos, means):
    gauss = gaussMd(mean, sigma=10)
    mean_RGBG, _ = raw.pull_apart(mean, colours)
    gauss_RGBG = gaussMd(mean_RGBG, sigma=(0,5,5))
    vmin, vmax = gauss_RGBG.min(), gauss_RGBG.max()

    plot.histogram_RGB(mean_RGBG, xlim=(xmin, xmax), xlabel="Bias (ADU)", saveto=savefolder/f"histogram_iso{iso}.pdf", nrbins=100)

    plot.show_image(gauss, colorbar_label="Bias (ADU)", saveto=savefolder/f"gauss_iso{iso}.pdf")
    for j, c in enumerate("RGBG"):
        X = "2" if j == 3 else ""
        plot.show_image(gauss_RGBG[j], colorbar_label="Bias (ADU)", saveto=savefolder/f"{c}{X}_gauss_iso{iso}.pdf", colour=c, vmin=vmin, vmax=vmax)

    plot.show_RGBG(gauss_RGBG, colorbar_label=25*" "+"Bias (ADU)", saveto=savefolder/f"all_gauss_iso{iso}.pdf", vmin=vmin, vmax=vmax)
