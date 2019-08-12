import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io

folders = io.path_from_input(argv)
plot_colours = ["black", "red", "xkcd:purple", "xkcd:olive", "xkcd:lilac", "xkcd:custard", "xkcd:peach"]

plt.figure(figsize=(4, 3), tight_layout=True)

xmax = 0

for c, folder in zip(plot_colours, folders):
    root, images, stacks, products, results = io.folders(folder)
    phone = io.load_metadata(root)

    iso_max = phone["software"]["ISO max"]

    products_iso = products/"iso"

    lookup_table = np.load(products/"iso_lookup_table.npy")
    data         = np.load(products/"iso_data.npy"        )

    plt.errorbar(data[0], data[1], yerr=data[2], fmt=f"o", c=c, label=phone["device"]["name"])
    plt.plot(*lookup_table, c=c)

    xmax = max(xmax, iso_max)

    print(phone["device"]["manufacturer"], phone["device"]["name"])

plt.xlabel("ISO speed")
plt.ylabel("Normalization")
plt.xlim(0, 2050)
plt.ylim(0, 30)
plt.grid(True)
plt.legend(loc="best")
plt.savefig(io.results_folder/"iso_comparison.pdf")
plt.show()
plt.close()
