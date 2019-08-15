"""
Analyse the ISO normalisation function for a single camera.

Command line arguments:
    * `folder`: the folder containing the ISO normalisation data to be 
    analysed.
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io

# Get the data folder from the command line
file = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(file)

# Get metadata
phone = io.load_metadata(root)
iso_max = phone["software"]["ISO max"]
save_to = root/"results/iso/iso_normalisation.pdf"

# Load the normalisation data and look-up table
lookup_table = np.load(products/"iso_lookup_table.npy")
data = np.load(products/"iso_data.npy")
print("Loaded data")

# Plot the normalisation data and look-up table
plt.figure(figsize=(4, 3), tight_layout=True)
plt.errorbar(data[0], data[1], yerr=data[2], fmt=f"o", c='k')
plt.plot(*lookup_table, c='k')
plt.xlabel("ISO speed")
plt.ylabel("Normalisation")
plt.xlim(0, iso_max*1.05)
plt.ylim(ymin=0)
plt.grid(True)
plt.savefig(save_to)
plt.close()
print(f"Saved plot to '{save_to}'")
