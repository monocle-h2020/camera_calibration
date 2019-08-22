import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import raw, plot, io, analyse, iso, calibrate
from spectacle.general import gaussMd

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
results_readnoise = results/"readnoise"

isos, stds  = io.load_stds  (folder, retrieve_value=io.split_iso)

lookup_table = iso.load_iso_lookup_table(root)

stds_mean = stds.mean(axis=(1,2))
stds_stds = stds.std (axis=(1,2))

plt.errorbar(isos, stds_mean, yerr=stds_stds, fmt="ko")
plt.xlabel("ISO speed")
plt.ylabel("Read noise (ADU)")
plt.xlim(0, lookup_table[0, -1]*1.05)
plt.ylim(ymin=0)
plt.savefig(results_readnoise/"iso_dependence.pdf")
plt.show()
plt.close()

stds_normalised = calibrate.normalise_iso(root, stds, isos)

stds_mean = stds_normalised.mean(axis=(1,2))
stds_stds = stds_normalised.std (axis=(1,2))

plt.errorbar(isos, stds_mean, yerr=stds_stds, fmt="ko")
plt.xlabel("ISO speed")
plt.ylabel("Read noise (norm. ADU)")
plt.xlim(0, lookup_table[0, -1]*1.05)
plt.ylim(ymin=0)
plt.savefig(results_readnoise/"normalised_iso_dependence.pdf")
plt.show()
plt.close()
