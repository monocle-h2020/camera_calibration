import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import io, gain

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")

iso_min = phone["software"]["ISO min"]
iso_max = phone["software"]["ISO max"]

results_gain = results/"gain"
isos, gainarrays = io.load_npy(folder, "iso*.npy", retrieve_value=io.split_iso)
gains, gain_errors = gainarrays.T

inverse_gains = 1/gains
inverse_gain_errors = inverse_gains**2 * gain_errors

model, model_error, model_label, parameters, covariances, R2 = gain.fit_iso_relation(isos, inverse_gains, inverse_gain_errors)

isorange = np.arange(iso_min, iso_max, 1)
inverse_gains_fit = model(isorange, *parameters)
inverse_gains_fit_errors = model_error(isorange, parameters, covariances)
gains_fit = 1 / inverse_gains_fit
gains_fit_errors = inverse_gains_fit_errors / inverse_gains_fit**2
lookup_table = np.stack([isorange, gains_fit, gains_fit_errors])
np.save(results_gain/"gain_lookup_table.npy", lookup_table)

inverse_gains_fit_data = model(isos, *parameters)
inverse_gains_fit_data_errors = model_error(isos, parameters, covariances)

xmaxes = [iso_max*1.01]
labels = [""]
if model is gain.model_knee:
    xmaxes.append(parameters[2]*1.5)
    labels.append("_zoom")

label_model, label_error = model_label(parameters, covariances)

for label, xmax in zip(labels, xmaxes):
    plt.figure(figsize=(7,5), tight_layout=True)
    plt.errorbar(isos, inverse_gains, yerr=inverse_gain_errors, fmt="o", c="k")
    plt.plot(isorange, inverse_gains_fit, c="k", label=label_model)
    plt.fill_between(isorange, inverse_gains_fit-inverse_gains_fit_errors, inverse_gains_fit+inverse_gains_fit_errors, color="0.5", label=label_error)
    plt.xlabel("ISO")
    plt.ylabel("$1/G$ (ADU/e$^-$)")
    plt.xlim(0, xmax)
    plt.ylim(ymin=0)
    plt.title(f"$R^2 = {R2:.4f}$")
    plt.legend(loc="lower right")
    plt.savefig(results_gain/f"iso_invgain{label}.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(7,5), tight_layout=True)
    plt.errorbar(isos, gains, yerr=gain_errors, fmt="o", c="k")
    plt.plot(isorange, gains_fit, c="k", label=label_model)
    plt.fill_between(isorange, gains_fit-gains_fit_errors, gains_fit+gains_fit_errors, color="0.5", label=label_error)
    plt.xlabel("ISO")
    plt.ylabel("$G$ (e$^-$/ADU)")
    plt.xlim(0, xmax)
    plt.ylim(ymin=0)
    plt.title(f"$R^2 = {R2:.4f}$")
    plt.legend(loc="upper right")
    plt.savefig(results_gain/f"iso_gain{label}.png")
    plt.show()
    plt.close()
