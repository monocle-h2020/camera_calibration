import numpy as np
from sys import argv
from phonecal import io, linearity as lin
from matplotlib import pyplot as plt

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")

normalisations, offsets, gammas, R2s = [np.load(results/f"linearity/{label_simple}.npy") for label_simple in ["normalization", "offset", "gamma", "R2"]]

for param, label, label_simple in zip([normalisations, offsets, gammas, R2s], ["Input normalization", "Offset (ADU)", "Gamma", "$R^2$"], ["normalization", "offset", "gamma", "R2"]):
    plt.figure(figsize=(4,2), tight_layout=True)
    for j, c in enumerate("rgb"):
        P = param[...,j].ravel()
        P = P[~np.isnan(P)]
        plt.hist(P, bins=250, color=c, alpha=0.7)
    plt.xlabel(label)
    plt.ylabel("Frequency")
    plt.xticks(rotation=20)
    plt.yscale("log")
    plt.title(f"{len(P)} pixels ({len(P)/(param.shape[0] * param.shape[1])*100:.1f}%)")
    plt.savefig(results/f"linearity/jpeg_{label_simple}.pdf")
    #plt.show()
    plt.close()
