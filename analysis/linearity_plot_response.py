"""
Plot the camera response at various incoming intensities for the central pixels
in a stack of images.

Command line arguments:
    * `folder`: the folder containing linearity data stacks. These should be
    NPY stacks taken at different exposure conditions, with the same ISO speed.
"""

import numpy as np
from sys import argv
from spectacle import io, plot, linearity as lin

# Get the data folder from the command line
folder = io.path_from_input(argv)
root = io.find_root_folder(folder)
savefolder = root/"analysis/linearity/"

# Load Camera object
camera = io.load_camera(root)
print(f"Loaded Camera object: {camera}")

# Find the indices of the central pixels
array_size = np.array(camera.image.shape)
mid1, mid2 = array_size // 2
center = np.s_[mid1:mid1+2, mid2:mid2+2]

# Bayer channels of the central pixels
colours_here = camera.bayer_map[center].ravel()

# Load the RAW data
intensities_with_errors, means = io.load_means(folder, retrieve_value=lin.filename_to_intensity, selection=center)
intensities, intensity_errors = intensities_with_errors.T
means = means.reshape((len(means), -1))
print("Loaded RAW data")

# Load the JPEG data, if available
try:
    intensities_with_errors, jmeans = io.load_jmeans(folder, retrieve_value=lin.filename_to_intensity, selection=center)
except ValueError:
    jpeg = False
else:
    intensities, intensity_errors = intensities_with_errors.T
    jpeg = True
    jmeans = jmeans.reshape((len(jmeans), -1, 3))
    print("Loaded JPEG data")

# Normalise the intensities
intensities = intensities / intensities.max()

if jpeg:
    plot.plot_linearity_dng_jpg(intensities, means, jmeans, colours_here, max_value=camera.saturation, savefolder=savefolder)
else:
    plot.plot_linearity_dng(intensities, means, colours_here, max_value=camera.saturation, savefolder=savefolder)
