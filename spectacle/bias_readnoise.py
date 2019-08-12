import numpy as np
from . import io


def load_bias_map(root):
    """
    Load the bias map located at `root`/products/bias.npy
    """
    bias_map = np.load(root/"products/bias.npy")
    return bias_map


def load_bias_metadata(root):
    """
    Load the bias value from the camera metadata file
    """
    metadata = io.load_metadata(root)
    bias_value = metadata["software"]["bias"]
    return bias_value
