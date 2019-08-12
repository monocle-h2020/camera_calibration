"""
Module for calibrating camera data based on calibration data.

If you are only interested in calibrating your data, using previously generated
calibrations, this is the module to use.
"""

from . import io
from .io import load_bias
from .iso import normalise_single_iso, normalise_multiple_iso

def correct_bias(root, data):
    """
    Perform a bias correction on data using a bias map from
    `root/products/bias.npy`.

    To do:
        - Use EXIF value if no map available
        - ISO selection
    """
    bias_map = load_bias(root)
    data_corrected = data - bias_map
    return data_corrected
