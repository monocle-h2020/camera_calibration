import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from ispex import raw, plot, io
from ispex.general import bin_centers
from glob import glob

folder = argv[1]
files = glob(f"{folder}/*.npy")

fitmin, fitmax = 750, 3750
nrmin = 2000

isos = np.unique([f.split("_")[1] for f in files]).astype(np.uint16)
isos.sort()

gains = np.zeros((len(isos), 3))
errs = gains.copy()
for i, iso in enumerate(isos):
    fig, axs = plt.subplots(2,3, sharex=True, sharey="row", figsize=(24, 8), tight_layout=True, gridspec_kw={"height_ratios": [4,1]})
    for j, c in enumerate("RGB"):
        try:
            bc, nr_per_I, mean_per_I, std_per_I = np.load(f"{folder}/iso_{iso}_{c}.npy")
        except FileNotFoundError:
            print(f"{folder}/iso_{iso}_{c}.npy does not exist")
            continue
        err_per_I = std_per_I/np.sqrt(nr_per_I-1)
        idx = np.where((nr_per_I > nrmin) & (bc >= fitmin) & (bc < fitmax))
        p,cov = np.polyfit(bc[idx], mean_per_I[idx], 1, w=np.sqrt(nr_per_I[idx]-1)/std_per_I[idx], cov=True)

        gains[i,j] = p[0]

        fitted = np.polyval(p, bc)
        diff = mean_per_I - fitted
        diffrel = diff / mean_per_I
        errs[i,j] = np.sqrt(np.mean(diffrel[idx]**2)) * gains[i,j]  # RMS

        ax = axs[0,j]
        ax.errorbar(bc, mean_per_I, yerr=err_per_I, color=c, fmt="o", label="Data", zorder=2)
        ax.plot(bc, fitted, ls="--", c="k", lw=2, label=f"$G \eta = {gains[i,j]:.2f} \pm {errs[i,j]:.2f}$", zorder=3)
        ax.legend(loc="lower right")

        derr_per_I = diffrel * err_per_I * np.sqrt(1/diff**2 + 1/err_per_I**2)

        ax = axs[1,j]
        ax.errorbar(bc, diffrel, yerr=derr_per_I, color=c, fmt="o", zorder=2)
        ax.plot(bc, np.zeros_like(bc), ls="--", c="k", lw=2, zorder=3)
    for ax in axs.ravel():
        ax.axvline(fitmin, lw=1, c='k')
        ax.axvline(fitmax, lw=1, c='k')
        ax.grid()
    ymax = gains[i].max() * 4096 * 1.05
    axs[0,0].set_ylim(0, ymax)
    axs[0,0].set_ylabel(r"$\sigma_C^2$")
    axs[1,0].set_ylabel(r"$\Delta_{\sigma^2} / \sigma^2$")
    axs[1,0].set_ylim(-1, 1)
    axs[1,0].set_xlabel(r"$\mu_C$")
    axs[0,0].set_xlim(0, 4096)
    fig.savefig(f"results/gain/Gain_iso_{iso}.png")
    plt.close()

for xmax, label in zip([1900, 250], ["", "_zoom"]):
    plt.figure(figsize=(8,5))
    for c, g, e in zip("RGB", gains.T, errs.T):
        plt.errorbar(isos, g, yerr=e, color=c, fmt="o")
    plt.xlabel("ISO")
    plt.ylabel(r"$G \eta$")
    plt.xlim(0, xmax)
    plt.ylim(0, 5)
    plt.grid()
    plt.tight_layout()
    plt.show()
    plt.close()
