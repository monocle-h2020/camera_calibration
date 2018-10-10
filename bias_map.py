import numpy as np
from sys import argv
from phonecal import io

folder = argv[1]
isos, means = io.load_means(folder, retrieve_value=io.split_iso, file=True)

low_iso = isos.argmin()
saveto = folder.replace("stacks", "products").strip("/")
np.save(f"{saveto}.npy", means[low_iso])
