"""
Code relating to dark current correction, such as fitting a trend or loading
a map.
"""

import numpy as np

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
    If `return_filename` is True, also return the exact filename the bias map
    was retrieved from.
    """
    filename = root/"calibration/dark_current_normalised.npy"
    dark_current_map = np.load(filename)
    if return_filename:
        return dark_current_map, filename
    else:
        return dark_current_map


def correct_dark_current_from_map(dark_current_map, data, exposure_time):
    """
    Apply a dark current correction from a dark current map `dark_current_map`,
    multiplied by an `exposure_time`, to an array `data`.
    """
    # Calculate the total dark current (in ADU) per pixel
    dark_total = dark_current_map * exposure_time

    # Correct the data
    data_corrected = data - dark_total

    return data_corrected
