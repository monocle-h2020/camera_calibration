import numpy as np
from sys import argv
from phonecal import io
from phonecal.gain import malus, malus_error
from phonecal.linearity import calculate_linear_R2_values

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")

angles,  means = io.load_means (folder, retrieve_value=io.split_pol_angle)
print("Read means")
colours        = io.load_colour(stacks)

offset_angle = io.load_angle(stacks)
print("Read angles")
intensities = malus(angles, offset_angle)
intensities_errors = malus_error(angles, offset_angle, sigma_angle0=1, sigma_angle1=1)

max_value = 2**phone["camera"]["bits"]
saturation = 0.95 * max_value

print("Doing R^2 comparison...", end=" ")

R2, saturated = calculate_linear_R2_values(intensities, means, saturate=saturation)

np.save(products/"linearity_R2.npy", R2)
