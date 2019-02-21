import numpy as np
from sys import argv
from phonecal import io, linearity as lin

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")

angles,  means = io.load_means (folder, retrieve_value=io.split_pol_angle)
print("Read means")

try:
    angles, jmeans = io.load_jmeans(folder, retrieve_value=io.split_pol_angle)
except ValueError:
    jpeg = False
    print("No JPEG data")
else:
    jpeg = True
    print("Read JPEG means")

colours = io.load_colour(stacks)

offset_angle = io.load_angle(stacks)
print("Read angles")
intensities = lin.malus(angles, offset_angle)
intensities_errors = lin.malus_error(angles, offset_angle, sigma_angle0=1, sigma_angle1=1)

max_value = 2**phone["camera"]["bits"]
saturation = 0.95 * max_value

print("Calculating Pearson r...", end=" ", flush=True)

r, saturated = lin.calculate_pearson_r_values(intensities, means, saturate=saturation)

print("... Done!")

np.save(products/"linearity_pearson_r.npy", r)

if jpeg:
    r_jpeg, saturated_jpeg = lin.calculate_pearson_r_values_jpeg(intensities, jmeans)
    np.save(products/"linearity_pearson_r_jpeg.npy", r_jpeg)