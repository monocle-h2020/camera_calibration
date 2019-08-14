import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io

file = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(file)
phone = io.load_metadata(root)
iso_max = phone["software"]["ISO max"]

lookup_table = np.load(products/"iso_lookup_table.npy")
data = np.load(products/"iso_data.npy")

plt.figure(figsize=(4, 3), tight_layout=True)
plt.errorbar(data[0], data[1], yerr=data[2], fmt=f"o", c='k')
plt.plot(*lookup_table, c='k')
plt.xlabel("ISO speed")
plt.ylabel("Normalization")
plt.xlim(0, iso_max*1.05)
plt.ylim(ymin=0)
plt.grid(True)
plt.savefig(root/"results/iso/iso_normalisation.pdf")
plt.show()
plt.close()
