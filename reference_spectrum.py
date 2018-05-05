from matplotlib import pyplot as plt
import numpy as np
from glob import glob
from sys import argv

folder = argv[1]
files = glob(folder+"/*.txt")

wavelength = np.loadtxt(files[0], skiprows=13, unpack=True)[0]
counts = np.array([np.loadtxt(f, skiprows=13, unpack=True)[1] for f in files])
counts = counts.T
counts[counts < 0] = 0

plt.figure(figsize=(10,5))
plt.plot(wavelength, counts)
plt.xlabel("$\lambda$ (nm)")
plt.xlim(350, 1000)
plt.ylabel("$I$ (counts)")
plt.ylim(-5, 4000)
plt.title(f"Halogen reference spectra ($N = {len(files)}$)")
plt.show()

counts_norm = counts / counts.max(axis=0)

plt.figure(figsize=(10,5))
plt.plot(wavelength, counts_norm)
plt.xlabel("$\lambda$ (nm)")
plt.xlim(350, 1000)
plt.ylabel("$I$ (normalised counts)")
plt.ylim(-0.05, 1.05)
plt.title(f"Halogen reference spectra ($N = {len(files)}$)")
plt.show()

counts_mean = counts_norm.mean(axis=1)
plt.figure(figsize=(10,5))
plt.plot(wavelength, counts_mean)
plt.xlabel("$\lambda$ (nm)")
plt.xlim(350, 1000)
plt.ylabel("$I$ (normalised counts)")
plt.ylim(-0.05, 1.05)
plt.title(f"Halogen reference spectra (mean)")
plt.show()

spec = np.hstack((wavelength[:,np.newaxis], counts_mean[:,np.newaxis]))
np.savetxt("spectrum_halogen.dat", spec, fmt="%i %f")