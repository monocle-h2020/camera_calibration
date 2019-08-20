"""
Compare two flat-field correction models.

Command line arguments:
    * `file1`: the location of the file containing the first model parameters.
    * `file2`: the location of the file containing the second model parameters.

To do:
    * Input labels for plots
"""

import numpy as np
from sys import argv
from spectacle import io, flat

# Get the data folder from the command line
file1, file2 = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(file1)

# Load the data
parameters1, errors1 = np.load(file1)
parameters2, errors2 = np.load(file2)

for param1, err1, param2, err2, label in zip(parameters1, errors1, parameters2, errors2, flat.parameter_labels):
    difference = param1 - param2
    difference_error = np.sqrt(err1**2 + err2**2)
    difference_sigma = abs(difference / difference_error)
    print(f"delta {label:>2} = {difference:+.4f} +- {difference_error:.4f} ({difference_sigma:>5.0f} sigma)")
