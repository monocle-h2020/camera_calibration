import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from ispex import raw, plot, io
from ispex.general import bin_centers, weighted_mean
from glob import glob

folder = argv[1]

files = glob(f"{folder}/*.npy")
isos = np.array([int(f.split("iso")[1].split(".")[0]) for f in files])
vals = np.array([np.load(f) for f in files])

#plt.figure(figsize=(8,6))
for xmax in (1850, 225):
    for j in range(3):
        plt.errorbar(isos, vals[:,0,j], yerr=vals[:,1,j], c="RGB"[j], fmt="o")
    plt.xlim(0, xmax)
    plt.ylim(0, 4.5)
    plt.xlabel("ISO")
    plt.ylabel("Gain")
    plt.grid()
    plt.show()
    plt.close()