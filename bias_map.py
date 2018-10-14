import numpy as np
from sys import argv
from phonecal import io

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
isos, means = io.load_means(folder, retrieve_value=io.split_iso)

low_iso = isos.argmin()
np.save(products/"bias.npy", means[low_iso])
