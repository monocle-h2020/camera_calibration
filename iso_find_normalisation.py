import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import io, plot, iso
from phonecal.general import Rsquare
from scipy.optimize import curve_fit

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")
min_iso = phone["software"]["ISO min"]
max_iso = phone["software"]["ISO max"]

results_iso = results/"iso"
print("Loaded information")

colours      = io.load_colour(stacks  )
bias         = io.load_bias  (products)
print("Loaded metadata")

isos, means = io.load_means (folder, retrieve_value=io.split_iso)
isos, stds  = io.load_stds  (folder, retrieve_value=io.split_iso)
print("Loaded data")

means -= bias

relative_errors = stds / means
median_relative_error = np.median(relative_errors)
print(f"Median relative error in photometry: {median_relative_error*100:.1f} %")

assert isos.min() == min_iso

ratios = means / means[isos.argmin()]
ratios_mean = ratios.mean(axis=(1,2))
ratios_errs = ratios.std (axis=(1,2))

model, R2 = iso.fit_iso_normalisation_relation(isos, ratios_mean, ratios_errs=ratios_errs, min_iso=min_iso, max_iso=max_iso)

iso_range = np.arange(0, max_iso+1, 1)
plt.figure(figsize=(3.3,3), tight_layout=True)
plt.errorbar(isos, ratios_mean, yerr=ratios_errs, fmt="ko", label="Data")
plt.plot(iso_range, model(iso_range), c='k', label=f"Fit")
plt.title(f"$R^2 = {R2:.6f}$")
plt.ylim(ymin=0)
plt.xlim(0, 1.01*phone["software"]["ISO max"])
plt.xlabel("ISO speed")
plt.ylabel("Normalization")
plt.legend(loc="lower right")
plt.savefig(results_iso/"normalization.pdf")
plt.show()
plt.close()

lookup_table = np.stack([iso_range, model(iso_range)])
data         = np.stack([isos, ratios_mean, ratios_errs])

np.save(products/"iso_lookup_table.npy", lookup_table)
np.save(products/"iso_data.npy"        , data)
