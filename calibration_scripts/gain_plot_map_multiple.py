import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import raw, io, plot
from spectacle.general import gauss_nan

files = io.path_from_input(argv)
roots = [io.folders(file)[0] for file in files]
cameras = [io.read_json(root/"info.json")["device"]["name"] for root in roots]
colours_arrays = [io.load_colour(root/"stacks") for root in roots]

isos = [io.split_iso(file) for file in files]

data_arrays = [np.load(file) for file in files]

for j, (c, label) in enumerate(zip("RGBG", [*"RGB", "G2"])):
    fig, axs = plt.subplots(ncols=len(files), figsize=(3*len(files), 2.3), squeeze=True, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0})
    for camera, iso, ax, data, colours in zip(cameras, isos, axs, data_arrays, colours_arrays):
        print(f"{camera:<10}: ISO {iso:>4}")
        RGBG,_ = raw.pull_apart(data, colours)
        gauss = gauss_nan(RGBG, sigma=(0,5,5))
        print(f"{c:>2}: {np.nanpercentile(RGBG, 0.1):.2f} -- {np.nanpercentile(RGBG, 99.9):.2f}")
        im = ax.imshow(gauss[j], cmap=plot.cmaps[c+"r"])
        ax.set_xticks([])
        ax.set_yticks([])
        if ax is axs[0]:
            loc = "left"
        elif ax is axs[-1]:
            loc = "right"
        else:
            loc = "bottom"
        cbar = plot.colorbar(im, location=loc, label="Gain (ADU/e$^-$)")
        ax.set_title(f"{camera} (ISO {iso})")
    fig.savefig(io.results_folder/f"gain_{label}.pdf")
    plt.show(fig)
    plt.close()

bins = np.linspace(0.4, 2.8, 250)
fig, axs = plt.subplots(ncols=len(files), nrows=3, figsize=(3*len(files), 2.3), tight_layout=True, gridspec_kw={"wspace":0, "hspace":0}, sharex=True, sharey=True)
for i, (camera, iso, ax_arr, data, colours) in enumerate(zip(cameras, isos, axs.T, data_arrays, colours_arrays)):
    RGBG,_ = raw.pull_apart(data, colours)
    R = RGBG[0].ravel()    ; R = R[~np.isnan(R)]
    G = RGBG[1::2].ravel() ; G = G[~np.isnan(G)]
    B = RGBG[2].ravel()    ; B = B[~np.isnan(B)]
    for ax, D, c in zip(ax_arr, [R, G, B], "rgb"):
        ax.hist(D, bins=bins, color=c, edgecolor=c, density=True)
        ax.grid(True)
        if i > 0:
            ax.tick_params(left=False)
    ax_arr[0] .set_title(f"{camera} (ISO {iso})")
    ax_arr[-1].set_xlabel("Gain (ADU/e$^-$)")
axs[1,0].set_ylabel("Frequency")
axs[0,0].set_xlim(bins[0], bins[-1])
axs[0,0].set_yticks([0.5, 1.5])
axs[0,0].set_xticks(np.arange(0.5, 3, 0.5))
fig.savefig(io.results_folder/"gain_hist.pdf")
plt.show(fig)
plt.close()

raise Exception
