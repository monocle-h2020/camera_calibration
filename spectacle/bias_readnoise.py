import numpy as np
from . import io


def load_bias_map(root, return_filename=False):
    """
    Load the bias map located at `root`/calibration/bias.npy.
    If `return_filename` is True, also return the exact filename the bias map
    was retrieved from.
    """
    filename = root/"calibration/bias.npy"
    bias_map = np.load(filename)
    if return_filename:
        return bias_map, filename
    else:
        return bias_map


def load_bias_metadata(root, return_filename=False):
    """
    Load the bias value from the camera metadata file
    If `return_filename` is True, also return the exact filename the metadata
    were retrieved from.
    TO DO:
        * Generate a bias map (Bayer tiling)
    """
    camera = io.load_metadata(root)
    bias_value = camera.image.bias
    return bias_value


def load_readnoise_map(root):
    """
    Load the bias map located at `root`/products/readnoise.npy
    """
    readnoise_map = np.load(root/"products/readnoise.npy")
    return readnoise_map
