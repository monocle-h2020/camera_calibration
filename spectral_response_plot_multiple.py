import numpy as np
from sys import argv
from spectacle import io, spectral
from spectacle.general import RMS
from matplotlib import pyplot as plt

files = io.path_from_input(argv)

folders = [io.folders(file)[0] for file in files]
cameras = [io.read_json(folder/"info.json")["device"]["name"] for folder in folders]

curves = [np.load(f) for f in files]

assert len(cameras) == len(curves)

number_of_cameras = len(cameras)

styles = ["-", "--", ":", "-."]

plt.figure(figsize=(7,3), tight_layout=True)
for i, (curve, camera, style) in enumerate(zip(curves, cameras, styles)):
    wavelength = curve[0]
    for j, c in enumerate("rgby"):
        mean  = curve[1+j]
        error = curve[5+j]
        over_20_percent = np.where(mean >= 0.2)[0]
        min_wvl, max_wvl = wavelength[over_20_percent[0]], wavelength[over_20_percent[-1]]
        print(f"{camera:>15} {c} RMS: {RMS(error):.3f}, >20% transmission at {min_wvl:.0f}-{max_wvl:.0f} nm, effective bandwidth {spectral.effective_bandwidth(wavelength, mean):>3.0f} nm")
        plt.plot(wavelength, mean, c=c, ls=style)
        #plt.fill_between(wavelength, mean-error, mean+error, color=c, alpha=0.5)
    plt.plot([-1000,-1001], [-1000,-1001], c='k', ls=style, label=camera)
    print(f"{camera:>15} RMS(G-G2) = {RMS(curve[2] - curve[4]):.4f}")
plt.grid(True)
plt.xticks(np.arange(0,1000,50))
plt.xlim(390, 700)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Relative sensitivity")
plt.ylim(0, 1.02)
plt.legend(loc="best")
plt.savefig(io.results_folder/"spectral_responses.pdf")
plt.show()
plt.close()

plt.figure(figsize=(7,3), tight_layout=True)
for i, (curve, camera, style) in enumerate(zip(curves, cameras, styles)):
    wavelength = curve[0]
    means = curve[1:5]
    errors = curve[5:]
    G = means[1::2].mean(axis=0)
    G_errors = 0.5 * np.sqrt((errors[1::2]**2).sum(axis=0))
    means_RGB = np.stack([means[0], G, means[2]])
    errors_RGB = np.stack([errors[0], G_errors, errors[2]])
    for j, c in enumerate("rgb"):
        mean  =  means_RGB[j]
        error = errors_RGB[j]
        over_20_percent = np.where(mean >= 0.2)[0]
        min_wvl, max_wvl = wavelength[over_20_percent[0]], wavelength[over_20_percent[-1]]
        print(f"{camera:>15} {c} RMS: {RMS(error):.3f}, >20% transmission at {min_wvl:.0f}-{max_wvl:.0f} nm, effective bandwidth {spectral.effective_bandwidth(wavelength, mean):>3.0f} nm")
        plt.plot(wavelength, mean, c=c, ls=style)
    plt.plot([-1000,-1001], [-1000,-1001], c='k',ls=style, label=camera)
plt.grid(True)
plt.xticks(np.arange(0,1000,50))
plt.xlim(390, 700)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Relative sensitivity")
plt.ylim(0, 1.02)
plt.legend(loc="best")
plt.savefig(io.results_folder/"spectral_responses_RGB.pdf")
plt.show()
plt.close()
