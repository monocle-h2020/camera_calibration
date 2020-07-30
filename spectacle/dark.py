"""
Code relating to dark current correction, such as fitting a trend or loading
a map.
"""

import numpy as np
from .general import return_with_filename, apply_to_multiple_args
from . import io

def fit_dark_current_linear(exposure_times, data):
    """
    Fit a linear trend to each pixel in the data set `data` taken at
    varying `exposure_times`.
    """

    # The data are reshaped to a flat list for each exposure time, fitted, and
    # then reshaped back to the original shape.
    original_shape = data.shape[1:]
    data_reshaped = data.reshape((data.shape[0], -1))

    # Both a bias (offset) and dark current (slope) are obtained from the
    # linear fit. The bias is treated as a free parameter to account for
    # variations and noise.
    dark_fit, bias_fit = np.polyfit(exposure_times, data_reshaped, 1)

    dark_reshaped = dark_fit.reshape(original_shape)
    bias_reshaped = bias_fit.reshape(original_shape)

    return dark_reshaped, bias_reshaped


def load_dark_current_map(root, return_filename=False):
    """
    Load the normalised dark current map located at root/`calibration/dark_current_normalised.npy`
    If `return_filename` is True, also return the exact filename used.
    """
    filename = io.find_files(root/"calibration", "dark_current_normalised.npy")
    dark_current_map = np.load(filename)
    return return_with_filename(dark_current_map, filename, return_filename)


def _correct_dark_current(data_element, dark_current, exposure_time):
    """
    Apply a dark current correction with value `dark_current` times the
    `exposure_time` to an element data_element
    """
    return data_element - dark_current * exposure_time


def correct_dark_current_from_map(dark_current_map, exposure_time, *data):
    """
    Apply a dark current correction from a dark current map `dark_current_map`,
    multiplied by an `exposure_time`, to any number of elements in `data`.
    """
    data_corrected = apply_to_multiple_args(_correct_dark_current, data, dark_current_map, exposure_time)

    return data_corrected
