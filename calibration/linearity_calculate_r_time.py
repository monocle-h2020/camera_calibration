import numpy as np
from sys import argv
from spectacle import io, linearity as lin

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.load_metadata(root)

times, means = io.load_means (folder, retrieve_value=io.split_exposure_time)
print("Read means")

colours      = io.load_colour(stacks)

max_value = 2**phone["camera"]["bits"]
saturation = 0.95 * max_value

print("Calculating Pearson r...", end=" ", flush=True)

r, saturated = lin.calculate_pearson_r_values(times, means, saturate=saturation)
del means
print("... Done!")

np.save(products/"linearity_pearson_r.npy", r)

try:
    times, jmeans = io.load_jmeans(folder, retrieve_value=io.split_exposure_time)
except ValueError:
    print("No JPEG data")
else:
    print("Read JPEG means")
    r_jpeg, saturated_jpeg = lin.calculate_pearson_r_values_jpeg(times, jmeans)
    np.save(products/"linearity_pearson_r_jpeg.npy", r_jpeg)
