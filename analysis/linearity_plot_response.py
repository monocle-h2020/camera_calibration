import numpy as np
from sys import argv
from spectacle import io, plot, linearity as lin

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.load_metadata(root)

colours = io.load_colour(stacks)

array_size = np.array(colours.shape)
mid1, mid2 = array_size // 2
center = np.s_[mid1:mid1+2, mid2:mid2+2]

colours_here = colours[center].ravel()

# Load the RAW data
intensities_with_errors, means = io.load_means(folder, retrieve_value=lin.filename_to_intensity, selection=center)
intensities, intensity_errors = intensities_with_errors.T
means = means.reshape((len(means), -1))
print("Loaded RAW data")

try:
    intensities_with_errors, jmeans = io.load_jmeans(folder, retrieve_value=lin.filename_to_intensity, selection=center)
    intensities, intensity_errors = intensities_with_errors.T
except ValueError:
    jpeg = False
else:
    jpeg = True
    jmeans = jmeans.reshape((len(jmeans), -1, 3))
    print("Loaded JPEG data")

intensities = intensities / intensities.max()

max_value = 2**phone["camera"]["bits"]

if jpeg:
    plot.plot_linearity_dng_jpg(intensities, means, jmeans, colours_here, max_value=max_value, savefolder=results/"linearity")
else:
    plot.plot_linearity_dng(intensities, means, colours_here, max_value=max_value, savefolder=results/"linearity")
