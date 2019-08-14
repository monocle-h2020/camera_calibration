import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import raw, io, plot
from spectacle.general import gauss_nan

file = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(file)
phone = io.load_metadata(root)

products_gain, results_gain = products/"gain", results/"gain"
ISO = io.split_iso(file)
colours = io.load_colour(stacks)
print("Loaded information")

gains = np.load(file)
gains_RGBG,_ = raw.pull_apart(gains, colours)
gains_gauss = gauss_nan(gains_RGBG, sigma=(0,5,5))

plot.histogram_RGB(gains_RGBG, xlim=(0,8), xlabel="Gain (ADU/e$^-$)", saveto=results_gain/f"hist_iso{ISO}.pdf")
print("Made histogram")

vmin, vmax = np.nanmin(gains_gauss), np.nanmax(gains_gauss)
plot.show_RGBG(gains_gauss, colorbar_label=25*" "+"Gain (ADU/e$^-$)", vmin=vmin, vmax=vmax, saveto=results_gain/f"map_iso{ISO}.pdf")
print("Made map")

gains_combined_gauss = gauss_nan(gains, sigma=10)
for gauss, label, cm in zip([gains_combined_gauss, *gains_gauss], ["combined", *"RGB", "G2"], [None, "Rr", "Gr", "Br", "Gr"]):
    plt.figure(figsize=(3.3,2.5), tight_layout=True)
    im = plt.imshow(gauss, cmap=plot.cmaps[cm])
    plt.xticks([])
    plt.yticks([])
    cbar = plot.colorbar(im)
    cbar.set_label("Gain (ADU/e-)")
    plt.savefig(results_gain/f"map_iso{ISO}_{label}.pdf")
    plt.close()
print("Made individual maps")

fig, axs = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(3.3,2.4), squeeze=True, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0})
axs[0].hist(gains_RGBG[0]   .ravel(), bins=np.linspace(0, 3.5, 250), color="r", edgecolor="r", density=True)
axs[1].hist(gains_RGBG[1::2].ravel(), bins=np.linspace(0, 3.5, 250), color="g", edgecolor="g", density=True)
axs[2].hist(gains_RGBG[2]   .ravel(), bins=np.linspace(0, 3.5, 250), color="b", edgecolor="b", density=True)
for ax in axs[:2]:
    ax.xaxis.set_ticks_position("none")
for ax in axs:
    ax.grid(True)
    ax.set_yticks([0,1,2])
axs[0].set_xlim(0, 3.5)
axs[0].set_ylim(0, 2.5)
axs[2].set_xlabel("Gain (ADU/e-)")
axs[1].set_ylabel("Frequency")
plt.savefig(results_gain/f"hist_iso{ISO}_RGB.pdf")
plt.close()
print("Made RGB histogram")
