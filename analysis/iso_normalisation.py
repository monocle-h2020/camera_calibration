"""
Analyse the ISO normalisation function for a single camera.

Command line arguments:
    * `folder`: the folders containing the ISO normalisation look-up tables
    and reduced data. These should be in NPY files generated using
    ../calibration/iso_normalisation.py.
"""

from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, iso

# Get the data folder from the command line
file = io.path_from_input(argv)
root = io.find_root_folder(file)
save_to = root/"analysis/iso_normalisation/iso_normalisation_curve.pdf"

# Load Camera object
camera = io.load_camera(root)
print(f"Loaded Camera object: {camera}")

# Load the normalisation data and look-up table
lookup_table = iso.load_iso_lookup_table(root)
data = iso.load_iso_data(root)
print("Loaded data")

# Plot the normalisation data and look-up table
plt.figure(figsize=(4, 3), tight_layout=True)
plt.errorbar(data[0], data[1], yerr=data[2], fmt=f"o", c='k')
plt.plot(*lookup_table, c='k')
plt.xlabel("ISO speed")
plt.ylabel("Normalisation")
plt.xlim(0, camera.settings.ISO_max * 1.05)
plt.ylim(ymin=0)
plt.grid(True)
plt.savefig(save_to)
plt.close()
print(f"Saved plot to '{save_to}'")
