"""
Module for calibrating camera data based on calibration data.

If you are only interested in calibrating your data, using previously generated
calibrations, this is the module to use.
"""

from . import io
from .iso import normalise_single_iso, normalise_multiple_iso

def correct_bias(root, data):
    """
    Perform a bias correction on data using a bias map from
    `root/products/bias.npy`.

    To do:
        - Use EXIF value if no map available
        - ISO selection
    """
    try:
        bias = io.load_bias_map(root)
    except FileNotFoundError:
        metadata = io.load_metadata(root)
        bias = metadata["software"]["bias"]
        print(f"Using bias value from metadata in `{root}/info.json`")
    else:
        print(f"Using bias map from `{root}/products/bias_map.npy`")
    data_corrected = data - bias
    return data_corrected
