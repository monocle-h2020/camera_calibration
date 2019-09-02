"""
Code relating to gain calibration, such as loading gain maps.
"""

import numpy as np

def load_gain_map(root, return_filename=False):
    """
    Load the gain map located at `root`/calibration/gain.npy

    If `return_filename` is True, also return the exact filename the bias map
    was retrieved from.
    """
    filename = root/"calibration/gain.npy"
    gain_map = np.load(filename)
    if return_filename:
        return gain_map, filename
    else:
        return gain_map
