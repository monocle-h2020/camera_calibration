import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, iso, plot

file = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(file)
phone = io.load_metadata(root)

results_gain = results/"gain"

ISO, gain_image = io.read_gain_table(file)
lookup_table = iso.load_iso_lookup_table(products)

midx, midy = np.array(gain_image.shape)[1:]//2

gain_per_iso = lookup_table[1] * gain_image[:, midx, midy][:, np.newaxis]

plt.figure(figsize=(4,4), tight_layout=True)
for j, c in enumerate(plot.rgbg):
    plt.scatter(ISO, gain_image[j, midx, midy], c=c, s=75)
    plt.plot(lookup_table[0], gain_per_iso[j], c=c)
plt.ylim(ymin=0)
plt.xlim(0, lookup_table[0,-1])
plt.xlabel("ISO speed")
plt.ylabel("Gain (ADU/e-)")
plt.show()
