import numpy as np
from sys import argv
from phonecal import io, linearity as lin
from matplotlib import pyplot as plt

folder, gamma = io.path_from_input(argv)
gamma = float(gamma.stem)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")

times, jmeans = io.load_jmeans(folder, retrieve_value=io.split_exposure_time)
print("Read means")

max_value = 2**phone["camera"]["bits"]
saturation = 0.95 * max_value

print(f"Gamma = {gamma}")
print("Fitting sRGB...")

normalizations, Rsquares, RMSes, RMSes_relative = lin.sRGB_compare_gamma(times, jmeans, gamma=gamma)

for param, label_simple in zip([normalizations, Rsquares, RMSes, RMSes_relative], ["normalization", "R2", "RMS", "RMS_rel"]):
    np.save(results/f"linearity/{label_simple}_gamma{gamma}.npy", param)
    print(f"Saved {label_simple}_gamma{gamma}.npy")
