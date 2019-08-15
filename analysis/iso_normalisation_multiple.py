"""
Analyse the ISO normalisation functions for multiple camera.

Command line arguments:
    * `folder`: the folders containing the ISO normalisation data 
    (multiple arguments possible)
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io

# Get the data folders from the command line
folders = io.path_from_input(argv)
save_to = io.results_folder/"iso_comparison.pdf"

# Colours to use for the different curves
plot_colours = ["black", "xkcd:lime green", "blue", "xkcd:peach"]

# Create the figure to hold the plot
plt.figure(figsize=(4, 3), tight_layout=True)

# Loop over each camera, load its data, and plot them in the figure
for c, folder in zip(plot_colours, folders):
    # Get the data folder for this camera
    root, images, stacks, products, results = io.folders(folder)
    
    # Get metadata
    phone = io.load_metadata(root)
    products_iso = products/"iso"

    # Load the normalisation data and look-up table
    lookup_table = np.load(products/"iso_lookup_table.npy")
    data = np.load(products/"iso_data.npy")

    # Plot the normalisation data and look-up table
    plt.errorbar(data[0], data[1], yerr=data[2], fmt=f"o", c=c, label=phone["device"]["name"])
    plt.plot(*lookup_table, c=c)

    print(phone["device"]["manufacturer"], phone["device"]["name"])

# Finish the plot with axes labels, legend, etc.
plt.xlabel("ISO speed")
plt.ylabel("Normalisation")
plt.xlim(0, 2050)
plt.ylim(0, 30)
plt.grid(True)
plt.legend(loc="best")
plt.savefig(save_to)
plt.close()
print(f"Saved plot to '{save_to}'")
