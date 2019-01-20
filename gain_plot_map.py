import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import raw, io, plot
from phonecal.general import gauss_nan

file = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(file)
phone = io.read_json(root/"info.json")

products_gain, results_gain = products/"gain", results/"gain"
ISO = io.split_iso(file)
colours = io.load_colour(stacks)
print("Loaded information")

gains = np.load(file)
gains_RGBG,_ = raw.pull_apart(gains, colours)
gains_gauss = gauss_nan(gains_RGBG, sigma=(0,5,5))

plot.hist_bias_ron_kRGB(gains_RGBG, xlim=(0,8), xlabel="Gain (ADU/e$^-$)", saveto=results_gain/f"hist_iso{ISO}.pdf")
print("Made histogram")

vmin, vmax = np.nanmin(gains_gauss), np.nanmax(gains_gauss)
plot.show_RGBG(gains_gauss, colorbar_label=25*" "+"Gain (ADU/e$^-$)", vmin=vmin, vmax=vmax, saveto=results_gain/f"map_iso{ISO}.pdf")
print("Made map")

gains_combined_gauss = gauss_nan(gains, sigma=10)

for gauss, label, cm in zip([gains_combined_gauss, *gains_gauss], ["combined", *"RGB", "G2"], [None, "Rr", "Gr", "Br", "Gr"]):
    plt.figure(figsize=(3.3,3), tight_layout=True)
    im = plt.imshow(gauss, cmap=plot.cmaps[cm])
    plt.xticks([])
    plt.yticks([])
    cbar = plot.colorbar(im)
    cbar.set_label("Gain (ADU/e-)")
    plt.savefig(results_gain/f"map_iso{ISO}_{label}.pdf")
    plt.close()
print("Made individual maps")
