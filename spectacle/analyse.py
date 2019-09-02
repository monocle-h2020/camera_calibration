"""
Module for common functions in analysing camera data or calibration data.
"""

from . import plot, raw
from .general import gaussMd, gauss_nan, symmetric_percentiles

import numpy as np
from astropy.table import Table
from matplotlib import pyplot as plt

def statistics(data, prefix_column=None, prefix_column_header=""):
    """
    Print statistics for the given data (list or array of arrays).

    Can include a prefix column to label data, such as ISO values or exposure
    times. Use `prefix_column` to pass these data. `prefix_column_format` is a
    format string describing how to print the `prefix_column`.
    `prefix_column_header` is the header to include for the prefix column.

    All other data are printed in `%.3f` format, i.e. float with 3 decimals.
    """

    # Calculate statistics for each element in `data`
    means               = np.mean  (data, axis=(1,2))
    medians             = np.median(data, axis=(1,2))
    standard_deviations = np.std   (data, axis=(1,2))
    maxes               = np.max   (data, axis=(1,2))
    mins                = np.min   (data, axis=(1,2))

    table_data = [means, medians, standard_deviations, maxes, mins]
    table_header = ["mean", "median", "std.dev.", "max", "min"]

    # Pre-pend the prefix column if one is given
    if prefix_column is not None:
        assert prefix_column_header not in table_header, f"Prefix column header `{prefix_column_header}` cannot be used because this is a default header name. Please use a different label."
        table_data = [prefix_column] + table_data
        table_header = [prefix_column_header] + table_header

    table = Table(data=table_data, names=table_header)

    if prefix_column is not None:
        table.sort(prefix_column_header)

    return table


def plot_gauss_maps(data, bayer_data, kernel_width_RGBG2=5, **kwargs):
    """
    Plot maps of the `data`, convolved with a Gaussian kernel. Both the
    mosaicked data and demosaicked RGBG2 data are convolved and plotted.
    The kernel width is `kernel_width_RGBG2` for the RGBG2 data and
    `2*kernel_width_RGBG2` for the mosaicked data.

    Any additional **kwargs are passed to both `plot.show_image` and
    `plot.show_image_RGBG2`.
    """

    # Determine which Gauss function to use
    if np.isnan(data).any():
        # Use gauss_nan if NaN values are present, since it can handle them
        gauss_function = gauss_nan
    else:
        # Use the faster gaussMd function if NaN values are not present
        gauss_function = gaussMd

    # Demosaick data by splitting the RGBG2 channels into separate arrays
    data_RGBG2,_ = raw.pull_apart(data, bayer_data)

    # Convolve the data with a Gaussian kernel
    # The two-dimensional mosaicked data are convolved over both axes
    # The three-dimensional demosaicked RGBG2 data are convolved over the two
    # spatial axes (1, 2), not the colour axis (0)
    kernel_width_mosaic = 2 * kernel_width_RGBG2
    data_gaussed = gauss_function(data, kernel_width_mosaic)
    data_RGBG2_gaussed = gauss_function(data_RGBG2, (0, kernel_width_RGBG2, kernel_width_RGBG2))

    plot.show_image(data_gaussed, **kwargs)
    plot.show_image_RGBG2(data_RGBG2_gaussed, **kwargs)


def plot_histogram_RGB(data, bayer_data, **kwargs):
    """
    Plot an RGB histogram of the `data`, demosaicked according to the
    `bayer_data`.

    Any additional **kwargs are passed to `plot.histogram_RGB`.
    """

    # Demosaick data by splitting the RGBG2 channels into separate arrays
    data_RGBG2,_ = raw.pull_apart(data, bayer_data)

    # Plot the RGB histogram
    plot.histogram_RGB(data_RGBG2, **kwargs)
