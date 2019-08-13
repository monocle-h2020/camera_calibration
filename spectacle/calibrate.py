"""
Module for calibrating camera data based on calibration data.

If you are only interested in calibrating your data, using previously generated
calibrations, this is the module to use.
"""

# Import other SPECTACLE submodules to use in functions
from . import bias_readnoise, dark, flat, io, iso

# Import functions from other SPECTACLE submodules which may be used in
# calibration scripts, for simpler access
from .bias_readnoise import load_bias_map, load_readnoise_map
from .dark import load_dark_current_map

def correct_bias(root, data):
    """
    Perform a bias correction on data using a bias map from
    `root`/products/bias.npy.

    To do:
        - Use EXIF value if no map available
        - ISO selection
    """
    try:
        bias = bias_readnoise.load_bias_map(root)
    except FileNotFoundError:
        bias = bias_readnoise.load_bias_metadata(root)
        print(f"Using bias value from metadata in `{root}/info.json`")
    else:
        print(f"Using bias map from `{root}/products/bias_map.npy`")
    data_corrected = data - bias

    return data_corrected


def correct_dark_current(root, data, exposure_time):
    """
    Perform a dark current correction on data using a dark current map from
    `root`/products/dark_normalised.npy

    To do:
        - Easy way to parse exposure times in scripts
    """
    dark_current = dark.load_dark_current_map(root)
    dark_total = dark_current * exposure_time
    data_corrected = data - dark_total

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


def correct_flatfield(root, data, shape):
    """
    Correction for flat-fielding using a flat-field model retrieved from
    `root`/products/flat_parameters.npy.

    To do:
        - Get shape from metadata
        - Choose between model and map (separate functions?)
    """
    correction_factor = flat.read_flat_field_correction(root, shape)
    data_corrected = data * correction_factor

    return data_corrected
