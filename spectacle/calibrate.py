"""
Module for calibrating camera data based on calibration data.

If you are only interested in calibrating your data, using previously generated
calibrations, this is the module to use.
"""

# Import other SPECTACLE submodules to use in functions
from . import bias_readnoise, dark, flat, gain, io, iso, spectral

# Import functions from other SPECTACLE submodules which may be used in
# calibration scripts, for simpler access
from .bias_readnoise import load_bias_map, load_readnoise_map
from .dark import load_dark_current_map
from .flat import clip_data, load_flatfield_correction
from .gain import load_gain_map
from .iso import load_iso_lookup_table
from .camera import load_camera
from .raw import demosaick
from .spectral import load_spectral_response, load_spectral_bandwidths, convert_RGBG2_to_RGB


def correct_bias(root, *data):
    """
    Perform a bias correction on data using a bias map from
    `root`/calibration/

    To do:
        - ISO selection
    """
    try:
        bias, origin = bias_readnoise.load_bias_map(root, return_filename=True)
    except FileNotFoundError:
        bias, origin = bias_readnoise.load_bias_metadata(root, return_filename=True)
        print(f"Using bias value from metadata in '{origin}'")
    else:
        print(f"Using bias map from '{origin}'")

    # Correct each given array
    data_corrected = bias_readnoise.correct_bias_from_map(bias, *data)

    return data_corrected


def correct_dark_current(root, exposure_time, *data):
    """
    Perform a dark current correction on data using a dark current map from
    `root`/calibration/

    To do:
        - Easy way to parse exposure times in scripts
    """
    # Load dark current map
    dark_current, origin = dark.load_dark_current_map(root, return_filename=True)
    print(f"Using dark current map from '{origin}'")

    # Correct each given array
    data_corrected = dark.correct_dark_current_from_map(dark_current, exposure_time, *data)

    return data_corrected


def normalise_iso(root, iso_values, *data):
    """
    Normalise data using an ISO normalisation look-up table from
    `root`/calibration/

    `iso_values` can be a single number (for a single ISO value) or a list-like object
    (for multiple)
    """
    lookup_table, origin = iso.load_iso_lookup_table(root, return_filename=True)
    print(f"Using ISO speed normalisation look-up table from '{origin}'")

    # Correct each given array
    data_corrected = iso.normalise_iso_general(lookup_table, iso_values, *data)

    return data_corrected


def convert_to_photoelectrons(root, *data):
    """
    Convert ISO-normalised data to photoelectrons using a normalised gain map
    (in normalised ADU per photoelectron) from `root`/calibration/
    """
    # Load the gain map
    gain_map, origin = gain.load_gain_map(root, return_filename=True)  # norm. ADU / e-
    print(f"Using normalised gain map from '{origin}'")

    # Correct each given array
    data_converted = gain.convert_to_photoelectrons_from_map(gain_map, *data)

    return data_converted


def correct_flatfield(root, *data, **kwargs):
    """
    Correction for flat-fielding using a flat-field correction map read from
    `root`/calibration/
    """
    # Load Camera object to get the array shape
    camera = load_camera(root)

    # Load the correction map
    correction_map, origin = flat.load_flatfield_correction(root, shape=camera.image.shape, return_filename=True)
    print(f"Using flat-field map from '{origin}'")

    # Correct each given array
    data_corrected = flat.correct_flatfield_from_map(correction_map, *data, **kwargs)

    return data_corrected


def correct_spectral_response(root, data_wavelengths, *data):
    """
    Correction for the spectral response of the camera, using curves read from
    `root`/calibration/

    The spectral responses are interpolated to the data_wavelengths given by the
    user. Spectral responses outside the range of the calibration data are
    assumed to be 0.

    The data are assumed to consist of 3 (RGB) or 4 (RGBG2) rows and a column
    for every wavelength. If not, an error is thrown.
    """
    # Load the spectral response curves
    spectral_response, origin = spectral.load_spectral_response(root, return_filename=True)
    print(f"Using spectral response curves from '{origin}'")

    # Normalise the input data by the spectral response and return the result
    data_normalised = spectral.correct_spectra(spectral_response, data_wavelengths, *data)

    return data_normalised
