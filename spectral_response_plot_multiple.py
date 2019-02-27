import numpy as np
from sys import argv
from phonecal import io
from phonecal.general import RMS
from matplotlib import pyplot as plt

files = io.path_from_input(argv)

folders = [io.folders(file)[0] for file in files]
cameras = [io.read_json(folder/"info.json")["device"]["name"] for folder in folders]

curves = [np.load(f) for f in files]

assert len(cameras) == len(curves)

number_of_cameras = len(cameras)

styles = ["-", "--", ":", ".-"]

plt.figure(figsize=(7,3), tight_layout=True)
for i, (curve, camera, style) in enumerate(zip(curves, cameras, styles)):
    wavelength = curve[0]
    for j, c in enumerate("rgby"):
        mean  = curve[1+j] / curve[1+j].max()
        error = curve[5+j] / curve[1+j].max()
        print(camera, c, f"RMS: {RMS(error):.3f}")
        plt.plot(wavelength, mean, c=c, ls=style)
        #plt.fill_between(wavelength, mean-error, mean+error, color=c, alpha=0.5)
    plt.plot([-1000,-1001], [-1000,-1001], c='k', ls=style, label=camera)
plt.grid(True)
plt.xticks(np.arange(0,1000,50))
plt.xlim(390, 700)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Relative sensitivity")
plt.ylim(0, 1.02)
plt.legend(loc="best")
plt.savefig("results/spectral_responses.pdf")
plt.show()
plt.close()

fig, axs = plt.subplots(nrows=number_of_cameras, sharex=True, sharey=True, figsize=(7,number_of_cameras*2), squeeze=True, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0})
for i, (curve, camera, ax) in enumerate(zip(curves, cameras, axs)):
    wavelength = curve[0]
    for j, c in enumerate("rgby"):
        mean  = curve[1+j] / curve[1+j].max()
        error = curve[5+j] / curve[1+j].max()
        ax.plot(wavelength, mean, c=c)
        ax.grid(True)
axs[0].set_xlim(390, 700)
axs[-1].set_xlabel("Wavelength (nm)")
axs[0].set_ylabel("Relative sensitivity")
plt.show()
