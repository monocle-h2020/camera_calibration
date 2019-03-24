import numpy as np
from sys import argv
from phonecal import io, linearity as lin
from matplotlib import pyplot as plt

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")

normalisations, gammas, R2s = [np.load(results/f"linearity/{label_simple}.npy") for label_simple in ["normalization", "gamma", "R2"]]

for param, label, label_simple in zip([normalisations, gammas, R2s], ["Input normalization", "Gamma", "$R^2$"], ["normalization", "gamma", "R2"]):
    xmin = np.nanpercentile(param.ravel(), 0.01)
    xmax = np.nanpercentile(param.ravel(), 99.9)
    bins = np.linspace(xmin, xmax, 250)
    for yscale in ["linear", "log"]:
        plt.figure(figsize=(4,2), tight_layout=True)
        for j, c in enumerate("rgb"):
            P = param[...,j].ravel()
            P = P[~np.isnan(P)]
            plt.hist(P, bins=bins, color=c, alpha=0.7)
        plt.xlabel(label)
        plt.ylabel("Frequency")
        plt.xlim(xmin, xmax)
        plt.xticks(rotation=20)
        plt.yscale(yscale)
        plt.title(f"{len(P)} pixels ({len(P)/(param.shape[0] * param.shape[1])*100:.1f}%)")
        plt.savefig(results/f"linearity/jpeg_{label_simple}_{yscale}.pdf")
        #plt.show()
        plt.close()

    print(label)
