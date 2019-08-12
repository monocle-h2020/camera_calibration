"""
Module for common functions in analysing camera data or calibration data.
"""

from . import plot

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
