import numpy as np
from sys import argv
from phonecal import io, linearity as lin

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")

times, means = io.load_means (folder, retrieve_value=io.split_exposure_time)
print("Read means")
colours      = io.load_colour(stacks)

max_value = 2**phone["camera"]["bits"]
saturation = 0.95 * max_value

print("Doing R^2 comparison...", end=" ", flush=True)

R2, saturated = lin.calculate_linear_R2_values(times, means, saturate=saturation)

print("... Done!")

np.save(products/"linearity_R2.npy", R2)
