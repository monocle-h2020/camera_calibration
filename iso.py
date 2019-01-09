import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import raw, io, plot
from phonecal.general import gaussMd, Rsquare

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")

products_iso, results_iso = products/"iso", results/"iso"
print("Loaded information")

colours      = io.load_colour(stacks  )
bias         = io.load_bias  (products)

isos, means = io.load_means (folder, retrieve_value=io.split_iso)
isos, stds  = io.load_stds  (folder, retrieve_value=io.split_iso)

means -= bias

relative_errors = stds / means
median_relative_error = np.median(relative_errors)
print(f"Median relative error in photometry: {median_relative_error*100:.1f} %")

assert isos.min() == phone["software"]["ISO min"]

ratios = means / means[isos.argmin()]
ratios_mean = ratios.mean(axis=(1,2))
ratios_errs = ratios.std (axis=(1,2))

ratios_RGBG,_ = raw.pull_apart(ratios, colours)

plt.errorbar(isos, ratios_mean, yerr=ratios_errs, fmt="ko")
plt.ylim(ymin=0)
plt.xlim(0, 1.01*phone["software"]["ISO max"])
plt.xlabel("ISO speed")
plt.ylabel("Normalization")
plt.show()
plt.close()
