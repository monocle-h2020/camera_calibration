import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import io, plot
from phonecal.linearity import malus, malus_error

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")

colours = io.load_colour(stacks)

array_size = io.array_size_dng(folder)
mid1, mid2 = array_size // 2
center = np.s_[mid1:mid1+2, mid2:mid2+2]

colours_here = colours[center].ravel()

angles, means = io.load_means (folder, retrieve_value=io.split_pol_angle, selection=center)
means = means.reshape((len(means), -1))
print("Loaded DNG data")

angles, jmeans = io.load_jmeans(folder, retrieve_value=io.split_pol_angle, selection=center)
jmeans = jmeans.reshape((len(jmeans), -1, 3))

offset_angle = io.load_angle(stacks)
print("Read angles")
intensities = malus(angles, offset_angle)
intensities_errors = malus_error(angles, offset_angle, sigma_angle0=1, sigma_angle1=1)

max_value = 2**phone["camera"]["bits"]

plot.plot_linearity_dng_jpg(intensities, means, jmeans, colours_here, intensities_errors=intensities_errors, max_value=max_value, savefolder=results/"linearity")
