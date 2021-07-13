"""
Code relating to bias and read noise, such as loading maps of either.
"""

import numpy as np
from . import io
from .general import return_with_filename


def load_bias_map(root, return_filename=False):
    """
    Load the bias map located at `root`/calibration/bias.npy.

    If `return_filename` is True, also return the exact filename used.
    """
    filename = io.find_matching_file(root/"calibration", "bias.npy")
    bias_map = np.load(filename)
    return return_with_filename(bias_map, filename, return_filename)


def load_bias_metadata(root, return_filename=False):
    """
    Load the bias value from the camera information file, and generate a Bayer-
    tiled map from it

    If `return_filename` is True, also return the exact filename used.
    """
    camera, filename = io.load_camera(root, return_filename=True)
    bias_map = camera.generate_bias_map()
    return return_with_filename(bias_map, filename, return_filename)


def load_readnoise_map(root, return_filename=False):
    """
    Load the bias map located at `root`/calibration/readnoise.npy

    If `return_filename` is True, also return the exact filename used.
    """
    filename = io.find_matching_file(root/"calibration", "readnoise.npy")
    readnoise_map = np.load(filename)
    return return_with_filename(readnoise_map, filename, return_filename)


def correct_bias_from_map(bias_map, data):
    """
    Apply a bias correction from a bias map `bias_map` to any number of
    elements in `data`
    """
    data_corrected = data - bias_map

    return data_corrected
