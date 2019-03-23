import numpy as np
from sys import argv
from phonecal import io, linearity as lin
from matplotlib import pyplot as plt

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")

angles, jmeans = io.load_jmeans(folder, retrieve_value=io.split_pol_angle)
print("Read means")

offset_angle = io.load_angle(stacks)
print("Read angles")
intensities = lin.malus(angles, offset_angle)
intensities_errors = lin.malus_error(angles, offset_angle, sigma_angle0=1, sigma_angle1=1)

max_value = 2**phone["camera"]["bits"]
saturation = 0.95 * max_value

print("Fitting sRGB...", end=" ", flush=True)

normalisations, offsets, gammas, R2s = lin.fit_sRGB_generic(intensities, jmeans)

for param, label, label_simple in zip([normalisations, offsets, gammas, R2s], ["Input normalization", "Offset (ADU)", "Gamma", "$R^2$"], ["normalization", "offset", "gamma", "R2"]):
    plt.figure(figsize=(4,4), tight_layout=True)
    for j, c in enumerate("rgb"):
        P = param[...,j].ravel()
        P = P[~np.isnan(P)]
        plt.hist(P, bins=250, color=c, alpha=0.7)
    plt.xlabel(label)
    plt.ylabel("Frequency")
    plt.title(f"{len(P)} pixels ({len(P)/(param.shape[0] * param.shape[1])*100:.1f}%)")
    plt.savefig(results/f"linearity/jpeg_{label_simple}.pdf")
    plt.show()
    plt.close()

    np.save(results/f"linearity/{label_simple}.npy", param)
