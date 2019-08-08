import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import raw, plot, io
from spectacle.general import gaussMd

files = io.path_from_input(argv)
stacks_folders = [io.folders(path)[2] for path in files]
colours = [io.load_colour(stacks) for stacks in stacks_folders]

data_all = [np.load(path) for path in files]
RGBGs_all = [raw.pull_apart(data, colour)[0] for data, colour in zip(data_all, colours)]
gauss_all = [gaussMd(RGBG, sigma=(0,5,5)) for RGBG in RGBGs_all]

vmin = min(gauss.min() for gauss in gauss_all)
vmax = max(gauss.max() for gauss in gauss_all)

fig, axs = plt.subplots(ncols=4, nrows=len(files), sharex=True, sharey=True, figsize=(7, 1.695*len(files)), squeeze=False, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0})

for path, gauss, axs_here in zip(files, gauss_all, axs):
    for j, (ax, D, c) in enumerate(zip(axs_here, gauss, "RGBG")):
        img = ax.imshow(D, cmap=plot.cmaps[c+"r"], vmin=vmin, vmax=vmax)
        ax.set_xticks([])
        ax.set_yticks([])

        if ax is axs[-1,j]:
            colorbar_here = plot.colorbar(img)
            if ax is axs_here[1]:
                colorbar_here.set_label(40*" " + "Read noise (ADU)")
            colorbar_here.locator = plot.ticker.MaxNLocator(nbins=3)
            colorbar_here.update_ticks()

path_image = io.results_folder/"RGBG_" + "_".join(path.stem for path in files) + ".pdf"
fig.savefig(path_image)
plt.show()
plt.close()
