import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from ispex import raw, plot, io
from ispex.general import bin_centers
from glob import glob

folder = argv[1]
files = glob(f"{folder}/*.npy")

isos = np.unique([f.split("_")[1] for f in files]).astype(np.uint16)

gains = np.zeros((len(isos), 3))
for i, iso in enumerate(isos):
    fig, axs = plt.subplots(3,1, sharex=True, sharey=True, figsize=(8, 24), tight_layout=True)
    for j, c in enumerate("RGB"):
        bc, nr_per_I, mean_per_I, std_per_I = np.load(f"{folder}/iso_{iso}_{c}.npy")
        idx = np.where((nr_per_I > 2000) & (bc < 3000))
        p = np.polyfit(bc[idx], mean_per_I[idx], 1, w=np.sqrt(nr_per_I[idx]-1)/std_per_I[idx])
        gains[i,j] = p[0]

        ax = axs.ravel()[j]
        ax.errorbar(bc, mean_per_I, yerr=std_per_I/np.sqrt(nr_per_I-1), color=c, fmt="o", label="Data", zorder=2)
        ax.plot(bc, np.polyval(p, bc), ls="--", c="k", lw=2, label=f"$G \eta = {gains[i,j]:.2f}$", zorder=3)
        ax.legend(loc="lower right")
        ax.grid()
        ax.set_ylabel(r"$\sigma_C^2$")
    ymax = gains[i].max() * 4096 * 1.05
    axs[0].set_ylim(0, ymax)
    axs[2].set_xlabel(r"$\mu_C$")
    axs[0].set_xlim(0, 4096)
    fig.savefig(f"results/gain/Gain_iso_{iso}.png")
    plt.close()