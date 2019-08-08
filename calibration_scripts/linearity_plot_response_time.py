import numpy as np
from sys import argv
from spectacle import io, plot, linearity as lin

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")

colours = io.load_colour(stacks)

array_size = io.array_size_dng(folder)
mid1, mid2 = array_size // 2
center = np.s_[mid1:mid1+2, mid2:mid2+2]

colours_here = colours[center].ravel()

times, means = io.load_means (folder, retrieve_value=io.split_exposure_time, selection=center)
means = means.reshape((len(means), -1))
print("Loaded DNG data")

try:
    times, jmeans = io.load_jmeans(folder, retrieve_value=io.split_exposure_time, selection=center)
except ValueError:
    jpeg = False
else:
    jpeg = True
    jmeans = jmeans.reshape((len(jmeans), -1, 3))
    print("Loaded JPEG data")

intensities = times / times.max()

max_value = 2**phone["camera"]["bits"]

if jpeg:
    plot.plot_linearity_dng_jpg(intensities, means, jmeans, colours_here, max_value=max_value, savefolder=results/"linearity")
else:
    plot.plot_linearity_dng(intensities, means, colours_here, max_value=max_value, savefolder=results/"linearity")
