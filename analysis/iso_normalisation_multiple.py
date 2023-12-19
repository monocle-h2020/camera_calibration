"""
Analyse the ISO normalisation functions for multiple camera.

Command line arguments:
    * `folder`: the folders containing the ISO normalisation look-up tables
    and reduced data. These should be in NPY files generated using
    ../calibration/iso_normalisation.py.
    (multiple arguments possible)
"""

from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, iso, plot

# Get the data folders from the command line
folders = io.path_from_input(argv)
save_to = io.results_folder/"iso_comparison.pdf"

# Plot colours
default_colours = plt.rcParams['axes.prop_cycle'].by_key()["color"]
colours = ["k", plot.RGB_OkabeIto[0], default_colours[4]]

# Create the figure to hold the plot
plt.figure(figsize=(5.33, 3), tight_layout=True)

# Loop over each camera, load its data, and plot them in the figure
for folder, colour in zip(folders, colours):
    # Get the data folder for this camera
    root = io.find_root_folder(folder)

    # Load Camera object
    camera = io.load_camera(root)
    print(f"Loaded Camera object: {camera}")

    # Load the normalisation data and look-up table
    lookup_table = iso.load_iso_lookup_table(root)
    data = iso.load_iso_data(root)

    # Plot the normalisation data and look-up table
    plt.errorbar(data[0], data[1], yerr=data[2], fmt="o", c=colour, label=camera.name)
    plt.plot(*lookup_table, c=colour)

    print(camera)

# Finish the plot with axes labels, legend, etc.
plt.xlabel("ISO speed")
plt.ylabel("Normalisation")
plt.xlim(0, 2050)
plt.ylim(0, 30)
plt.grid(True)
plt.legend(loc="best", framealpha=1, edgecolor="k", handletextpad=0.5)
plt.savefig(save_to)
plt.close()
print(f"Saved plot to '{save_to}'")
