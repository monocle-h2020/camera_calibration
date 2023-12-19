"""
Code relating to gain calibration, such as loading gain maps.
"""

import numpy as np

from . import io
from .general import return_with_filename


def load_gain_map(root, return_filename=False):
    """
    Load the gain map located at `root`/calibration/gain.npy

    If `return_filename` is True, also return the exact filename used.
    """
    filename = io.find_matching_file(root/"calibration", "gain.npy")
    gain_map = np.load(filename)
    return return_with_filename(gain_map, filename, return_filename)


def convert_to_photoelectrons_from_map(gain_map, data):
    """
    Convert `data` from normalised ADU to photoelectrons using a map of gain
    in each pixel `gain_map`.
    """
    data_converted = data / gain_map

    return data_converted
