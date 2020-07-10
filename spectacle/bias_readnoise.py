"""
Code relating to bias and read noise, such as loading maps of either.
"""

import numpy as np
from . import io


def load_bias_map(root, return_filename=False):
    """
    Load the bias map located at `root`/calibration/bias.npy.
    If `return_filename` is True, also return the exact filename used.
    """
    filename = root/"calibration/bias.npy"
    bias_map = np.load(filename)
    if return_filename:
        return bias_map, filename
    else:
        return bias_map


def load_bias_metadata(root, return_filename=False):
    """
    Load the bias value from the camera metadata file, and generate a Bayer-
    tiled map from it
    If `return_filename` is True, also return the exact filename used.
    """
    camera, filename = io.load_metadata(root, return_filename=True)
    bias_map = camera.generate_bias_map()
    if return_filename:
        return bias_map, filename
    else:
        return bias_map


def load_readnoise_map(root, return_filename=False):
    """
    Load the bias map located at `root`/calibration/readnoise.npy
    If `return_filename` is True, also return the exact filename used.
    """
    filename = root/"calibration/readnoise.npy"
    readnoise_map = np.load(filename)
    if return_filename:
        return readnoise_map, filename
    else:
        return readnoise_map


def correct_bias_from_map(bias_map, *data):
    """
    Apply a bias correction from a bias map `bias_map` to arrays `data` (any number)
    """

    # Not casting *data to a numpy array because it may have multiple shapes
    data_corrected = [data_array - bias_map for data_array in data]

    # If only a single array was given, don't return a list
    if len(data_corrected) == 1:
        data_corrected = data_corrected[0]

    return data_corrected
