"""
Analyse the ISO normalisation functions for multiple camera.

Command line arguments:
    * `folder`: the folders containing the ISO normalisation data
    (multiple arguments possible)
"""

from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, plot, iso

# Get the data folders from the command line
folders = io.path_from_input(argv)
save_to = io.results_folder/"iso_comparison.pdf"

# Create the figure to hold the plot
plt.figure(figsize=(4, 3), tight_layout=True)

# Loop over each camera, load its data, and plot them in the figure
for c, folder in zip(plot.line_colours, folders):
    # Get the data folder for this camera
    root = io.find_root_folder(folder)

    # Get metadata
    camera = io.load_metadata(root)

    # Load the normalisation data and look-up table
    lookup_table = iso.load_iso_lookup_table(root)
    data = iso.load_iso_data(root)

    # Plot the normalisation data and look-up table
    plt.errorbar(data[0], data[1], yerr=data[2], fmt=f"o", c=c, label=camera.device.name)
    plt.plot(*lookup_table, c=c)

    print(camera)

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
