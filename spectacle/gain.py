"""
Code relating to gain calibration, such as loading gain maps.
"""

import numpy as np
from .general import return_with_filename, apply_to_multiple_args

def load_gain_map(root, return_filename=False):
    """
    Load the gain map located at `root`/calibration/gain.npy

    If `return_filename` is True, also return the exact filename used.
    """
    filename = root/"calibration/gain.npy"
    gain_map = np.load(filename)
    return return_with_filename(gain_map, filename, return_filename)


def _convert_to_photoelectrons(data_element, gain_map):
    """
    Convert a `data_element` from ADU to e- using the `gain_map`
    """
    return data_element / gain_map


def convert_to_photoelectrons_from_map(gain_map, *data):
    """
    Convert `data` from normalised ADU to photoelectrons using a map of gain
    in each pixel `gain_map`.
    """
    data_converted = apply_to_multiple_args(_convert_to_photoelectrons, data, gain_map)

    return data_converted
