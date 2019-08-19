import numpy as np
from sys import argv
from spectacle import io, linearity as lin

folder, gamma = io.path_from_input(argv)
gamma = float(str(gamma))
root, images, stacks, products, results = io.folders(folder)
phone = io.load_metadata(root)

intensities_with_errors, jmeans = io.load_jmeans(folder, retrieve_value=lin.filename_to_intensity)
intensities, intensity_errors = intensities_with_errors.T
print("Read means")

max_value = 2**phone["camera"]["bits"]
saturation = 0.95 * max_value

print(f"Gamma = {gamma}")
print("Fitting sRGB...")

normalizations, Rsquares, RMSes, RMSes_relative = lin.sRGB_compare_gamma(intensities, jmeans, gamma=gamma)

for param, label_simple in zip([normalizations, Rsquares, RMSes, RMSes_relative], ["normalization", "R2", "RMS", "RMS_rel"]):
    np.save(results/f"linearity/{label_simple}_gamma{gamma}.npy", param)
    print(f"Saved {label_simple}_gamma{gamma}.npy")
