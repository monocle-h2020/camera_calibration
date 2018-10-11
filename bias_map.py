import numpy as np
from sys import argv
from phonecal import io

folder = argv[1]
root, images, stacks, products, results = io.folders(folder)
isos, means = io.load_means(folder, retrieve_value=io.split_iso, file=True)

low_iso = isos.argmin()
np.save(f"{products}/bias.npy", means[low_iso])
