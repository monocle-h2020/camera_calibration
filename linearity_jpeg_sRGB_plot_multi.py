import numpy as np
from sys import argv
from spectacle import io, linearity as lin
from matplotlib import pyplot as plt

files = io.path_from_input(argv)
roots = [io.folders(file)[0] for file in files]
cameras = [io.read_json(root/"info.json")["device"]["name"] for root in roots]
print("Read meta-data")

gammas_best = [np.load(root/"results/linearity/gamma.npy") for root in roots]
RMS_rel_22 = [np.load(root/"results/linearity/RMS_rel_gamma2.2.npy") for root in roots]
RMS_rel_24 = [np.load(root/"results/linearity/RMS_rel_gamma2.4.npy") for root in roots]
print("Read data")

gamma_bins = np.linspace(1.75, 2.65, 1000)
RMS_rel_bins = np.linspace(0, 34, 500)
bins = [gamma_bins, RMS_rel_bins, RMS_rel_bins]

titles = ["", r"$\gamma = 2.2$", r"$\gamma = 2.4$"]
xlabels = [r"Best fit $\gamma$", "RMS diff. (%)", "RMS diff. (%)"]

print("Generating plot...")
fig, axs_table = plt.subplots(ncols=3, nrows=len(cameras), tight_layout=True, sharex="col", sharey="row", figsize=(6, 1.4*len(cameras)), gridspec_kw={"wspace":0, "hspace":0})
for axs_row, camera, gamma, RMS22, RMS24 in zip(axs_table, cameras, gammas_best, RMS_rel_22, RMS_rel_24):
    print(camera)
    RMS22 = RMS22.copy() * 100 ; RMS24 = RMS24.copy() * 100  # convert to percentage
    print(f"RMS for gamma 2.2: {np.nanpercentile(RMS22.ravel(), 0.1):.0f} -- {np.nanpercentile(RMS22.ravel(), 99.9):.0f}")
    print(f"RMS for gamma 2.4: {np.nanpercentile(RMS24.ravel(), 0.1):.0f} -- {np.nanpercentile(RMS24.ravel(), 99.9):.0f}")
    for ax, param, b in zip(axs_row, [gamma, RMS22, RMS24], bins):
        for j, c in enumerate("rgb"):
            P = param[..., j].ravel()
            P = P[~np.isnan(P)]
            ax.hist(P, bins=b, color=c, edgecolor="none", alpha=0.7)
    axs_row[0].set_ylabel(camera)
    axs_row[0].set_ylim(ymin=0)
print("Setting plot parameters")
for title, ax in zip(titles, axs_table[0]):
    ax.set_title(title)
for xlabel, ax in zip(xlabels, axs_table[-1]):
    ax.set_xlabel(xlabel)
for b, ax in zip(bins, axs_table[-1].ravel()):
    ax.set_xlim(b.min(), b.max())
for ax in axs_table[:,1:].ravel():
    ax.tick_params(axis="y", left=False)
for ax in axs_table[:,0].ravel():
    ax.locator_params(axis="y", nbins=4)
for ax in axs_table[-1].ravel():
    ax.locator_params(axis="x", nbins=4)
plt.savefig("results/jpeg_gamma.pdf")
print("Saved plot")
plt.show()
plt.close()
