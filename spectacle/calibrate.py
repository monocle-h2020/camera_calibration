"""
Module for calibrating camera data based on calibration data.

If you are only interested in calibrating your data, using previously generated
calibrations, this is the module to use.
"""

from . import io, iso

def correct_bias(root, data):
    """
    Perform a bias correction on data using a bias map from
    `root`/products/bias.npy.

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


def normalise_iso(root, data, iso_values):
    """
    Normalise data using an ISO normalisation look-up table from
    `root`/products/iso_lookup_table.npy.

    If `iso` is a single number, use `normalise_single_iso`. Otherwise, use
    `normalise_multiple_iso`.
    """
    lookup_table = iso.load_iso_lookup_table(root)

    if isinstance(iso_values, (int, float)):
        data_normalised = iso.normalise_single_iso  (data, iso_values, lookup_table)
    else:
        data_normalised = iso.normalise_multiple_iso(data, iso_values, lookup_table)

    return data_normalised
