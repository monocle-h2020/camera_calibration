import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import io, gain

folders = io.path_from_input(argv)
colours = ["k", "r", "xkcd:purple", "xkcd:brown", "xkcd:lilac", "xkcd:custard"]

plt.figure(figsize=(7,5), tight_layout=True)

xmax = 0

for c, folder in zip(colours, folders):
    root, images, stacks, products, results = io.folders(folder)
    phone = io.read_json(root/"info.json")

    iso_min = phone["software"]["ISO min"]
    iso_max = phone["software"]["ISO max"]

    results_gain = results/"gain"
    isos, gainarrays = io.load_npy(products/"gain", "iso*.npy", retrieve_value=io.split_iso)
    gains, gain_errors = gainarrays.T

    inverse_gains = 1/gains
    inverse_gain_errors = inverse_gains**2 * gain_errors
    
    iso_lut, gain_lut, gain_err_lut = np.load(results/"gain/gain_lookup_table.npy")
    invgain_lut = 1/gain_lut
    #invgain_err_lut = 
    print(iso_lut.min(), iso_lut.max())
    
    plt.errorbar(isos, inverse_gains, yerr=inverse_gain_errors, fmt=f"o", c=c)
    plt.plot(iso_lut, invgain_lut, label=phone["device"]["name"], c=c)
    #plt.fill_between(iso_lut, inverse_gains_fit-inverse_gains_fit_errors, inverse_gains_fit+inverse_gains_fit_errors, color="0.5")
    
    xmax = max(xmax, iso_lut.max())

plt.xlabel("ISO")
plt.ylabel("$1/G$ (ADU/e$^-$)")
plt.xlim(0, xmax)
plt.ylim(ymin=0)
plt.legend(loc="lower right")
plt.savefig("gain_comparison.png")
plt.show()
plt.close()
