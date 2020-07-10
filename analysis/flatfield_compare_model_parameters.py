"""
Compare two flat-field correction models.

Command line arguments:
    * `file1`: the location of the file containing the first model parameters.
    * `file2`: the location of the file containing the second model parameters.
    These flat-field models should be in CSV files generated using
    ../calibration/flatfield.py

To do:
    * Input labels for plots
"""

import numpy as np
from sys import argv
from spectacle import io, flat

# Get the data folder from the command line
file1, file2 = io.path_from_input(argv)
root = io.find_root_folder(file1)

# Load the data
data1 = np.loadtxt(file1, delimiter=",")
parameters1, errors1 = data1[:7], data1[7:]
data2 = np.loadtxt(file2, delimiter=",")
parameters2, errors2 = data2[:7], data2[7:]

for param1, err1, param2, err2, label in zip(parameters1, errors1, parameters2, errors2, flat.parameter_labels):
    difference = param1 - param2
    difference_error = np.sqrt(err1**2 + err2**2)
    difference_sigma = abs(difference / difference_error)
    print(f"delta {label:>2} = {difference:+.4f} +- {difference_error:.4f} ({difference_sigma:>5.0f} sigma)")
