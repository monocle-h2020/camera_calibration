import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import io
from phonecal.general import Rsquare
from scipy.optimize import curve_fit

folder = argv[1]
isos, gainarrays = io.load_npy(folder, "iso*.npy", retrieve_value=io.split_iso, file=True)
gains, gain_errors = gainarrays.T

inverse_gains = 1/gains
inverse_gain_errors = inverse_gains**2 * gain_errors

def model(iso, slope, offset, knee):
    results = np.tile(knee * slope + offset, len(iso))
    results[iso < knee] = iso[iso < knee] * slope + offset
    return results

def model_error(iso, popt, pcov):
    results = np.tile(popt[2]**2 * pcov[0,0] + popt[0]**2 * pcov[2,2] + pcov[1,1], len(iso))
    results[iso < popt[2]] = iso[iso < popt[2]]**2 * pcov[0,0] + pcov[1,1]
    results = np.sqrt(results)
    return results

popt, pcov = curve_fit(model, isos, inverse_gains, p0=[0.1, 0.1, 200], sigma=inverse_gain_errors)

isorange = np.arange(0, 2000, 1)
inverse_gains_fit = model(isorange, *popt)
inverse_gains_fit_errors = model_error(isorange, popt, pcov)
gains_fit = 1 / inverse_gains_fit
gains_fit_errors = inverse_gains_fit_errors / inverse_gains_fit**2
lookup_table = np.stack([isorange, gains_fit, gains_fit_errors])
np.save("results/gain_new/gain_lookup_table.npy", lookup_table)

inverse_gains_fit_data = model(isos, *popt)
inverse_gains_fit_data_errors = model_error(isos, popt, pcov)
R2 = Rsquare(inverse_gains, inverse_gains_fit_data)

for xmax in (1850, 250):
    plt.figure(figsize=(7,5), tight_layout=True)
    plt.errorbar(isos, inverse_gains, yerr=inverse_gain_errors, fmt="o", c="k")
    plt.plot(isorange, inverse_gains_fit, c="k", label=f"slope: {popt[0]:.4f}\noffset: {popt[1]:.4f}\nknee: {popt[2]:.1f}")
    plt.fill_between(isorange, inverse_gains_fit-inverse_gains_fit_errors, inverse_gains_fit+inverse_gains_fit_errors, color="0.5", label=f"$\sigma$ slope: {np.sqrt(pcov[0,0]):.4f}\n$\sigma$ offset: {np.sqrt(pcov[1,1]):.4f}\n$\sigma$ knee: {np.sqrt(pcov[2,2]):.1f}")
    plt.xlabel("ISO")
    plt.ylabel("$1/G$ (ADU/e$^-$)")
    plt.xlim(0, xmax)
    plt.ylim(0, 5)
    plt.title(f"$R^2 = {R2:.4f}$")
    plt.legend(loc="lower right")
    plt.savefig(f"results/gain_new/iso_invgain_{xmax}.png")
    plt.show()
    plt.close()

    plt.figure(figsize=(7,5), tight_layout=True)
    plt.errorbar(isos, gains, yerr=gain_errors, fmt="o", c="k")
    plt.plot(isorange, gains_fit, c="k", label=f"slope: {popt[0]:.4f}\noffset: {popt[1]:.4f}\nknee: {popt[2]:.1f}")
    plt.fill_between(isorange, gains_fit-gains_fit_errors, gains_fit+gains_fit_errors, color="0.5",
                     label=f"$\sigma$ slope: {np.sqrt(pcov[0,0]):.4f}\n$\sigma$ offset: {np.sqrt(pcov[1,1]):.4f}\n$\sigma$ knee: {np.sqrt(pcov[2,2]):.1f}")
    plt.xlabel("ISO")
    plt.ylabel("$G$ (e$^-$/ADU)")
    plt.xlim(0, xmax)
    plt.ylim(0, 5)
    plt.title(f"$R^2 = {R2:.4f}$")
    plt.legend(loc="upper right")
    plt.savefig(f"results/gain_new/iso_gain_{xmax}.png")
    plt.show()
    plt.close()
