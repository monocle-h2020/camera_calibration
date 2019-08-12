import numpy as np
from sys import argv
from spectacle import io, linearity as lin
from matplotlib import pyplot as plt

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.load_metadata(root)

times, jmeans = io.load_jmeans(folder, retrieve_value=io.split_exposure_time)
print("Read means")

max_value = 2**phone["camera"]["bits"]
saturation = 0.95 * max_value

print("Fitting sRGB...")

normalisations, gammas, R2s = lin.fit_sRGB_generic(times, jmeans)

for param, label_simple in zip([normalisations, gammas, R2s], ["normalization", "gamma", "R2"]):
    np.save(results/f"linearity/{label_simple}.npy", param)
    print(f"Saved {label_simple}.npy")
