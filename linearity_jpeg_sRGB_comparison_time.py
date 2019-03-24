import numpy as np
from sys import argv
from phonecal import io, linearity as lin
from matplotlib import pyplot as plt

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")

times, jmeans = io.load_jmeans(folder, retrieve_value=io.split_exposure_time)
print("Read means")

max_value = 2**phone["camera"]["bits"]
saturation = 0.95 * max_value

print("Fitting sRGB...")

gammas = [2.2, 2.4]
normalizations, Rsquares, RMSes, RMSes_relative = lin.sRGB_compare_gammas(times, jmeans, gammas=gammas)

for g, gamma in gammas:
    print(gamma)
    for param, label_simple in zip([normalizations, Rsquares, RMSes, RMSes_relative], ["normalization", "R2", "RMS", "RMS_rel"]):
        p = param[g]
        np.save(results/f"linearity/{label_simple}_gamma{gamma}.npy", p)
        print(f"Saved {label_simple}.npy")
