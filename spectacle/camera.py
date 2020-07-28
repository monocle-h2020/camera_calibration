"""
Code relating to camera metadata, such as generating or reading metadata files.
"""

import numpy as np
import json
from collections import namedtuple
from pathlib import Path

from . import raw, analyse, bias_readnoise, dark, iso, gain, flat, spectral
from .general import return_with_filename


def find_root_folder(input_path):
    """
    For a given `input_path`, find the root folder, containing the standard
    sub-folders (calibration, analysis, stacks, etc.)
    """
    input_path = Path(input_path)

    # Loop through the input_path's parents until a metadata JSON file is found
    for parent in [input_path, *input_path.parents]:
        # If a metadata file is found, use the containing folder as the root folder
        if (parent/"camera.json").exists():
            root = parent
            break
    # If no metadata file was found, raise an error
    else:
        raise OSError(f"None of the parents of the input `{input_path}` include a 'camera.json' file.")

    return root


def name_from_root_folder(root):
    """
    Convert the root path to a camera name by removing underscores.
    This is useful for cameras that use model numbers internally,
    such as the Samsung Galaxy S8 (SM-G950F).
    """
    root = Path(root)
    stem = root.stem
    clean = stem.replace("_", " ").strip()
    return clean


def _convert_exposure_time(exposure):
    """
    Convert an exposure time, in various formats, into a floating-point number.
    """
    if isinstance(exposure, (float, int)):
        # If the input is already a number, simply return it
        return exposure
    elif isinstance(exposure, str):
        # If the input is a string, e.g. from the command line
        if "/" in exposure:
            # If a fraction (e.g. '1/3', '2/5.1') is given, simply evaluate
            # it and return the result
            num, den = [float(x) for x in exposure.split("/")]
            return num/den

        elif "." in exposure:
            # If a decimal number (e.g. '2.5' or '0.002') is given, simply
            # convert it to a floating point number and return the result
            return float(exposure)

        else:
            # If a simple number is given (e.g. '2' or '100'), simply convert
            # it to a floating point number and return the result
            return float(exposure)
    else:
        # If the input is none of the above, try to cast it to a float somehow
        try:
            exposure_new = float(exposure)
        except TypeError:
            # Raise an error if the exposure could not be converted
            raise TypeError(f"Input exposure `{exposure}` is of type `{type(exposure)}` which cannot be converted to a number.")
        else:
            # If the exposure could be converted, return it
            return exposure_new


class Camera(object):
    """
    Object that represents a camera, storing its important properties and providing
    functions for calibrating data.
    """
    # Properties a Camera can have
    property_list = ["name", "manufacturer", "name_internal", "image_shape", "raw_extension", "bias", "bayer_pattern", "bit_depth", "colour_description"]
    Settings = namedtuple("Settings", ["ISO_min", "ISO_max", "exposure_min", "exposure_max"])

    calibration_data_all = ["bias_map", "readnoise", "dark_current", "iso_lookup_table", "gain_map", "flatfield_map", "spectral_response"]

    def __init__(self, name, manufacturer, name_internal, image_shape, raw_extension, bias, bayer_pattern, bit_depth, colour_description="RGBG", root=None):
        """
        Generate a Camera object based on input dictionaries containing the
        keys defined in Device and Image

        If a root folder is provided, this will be used to load calibration data from.
        """
        # Save properties
        self.name = name
        self.name_underscore = self.name.replace(" ", "_")
        self.manufacturer = manufacturer
        self.name_internal = name_internal
        self.image_shape = image_shape
        self.raw_extension = raw_extension
        self.bias = bias
        self.bayer_pattern = bayer_pattern
        self.bit_depth = bit_depth
        self.colour_description = colour_description
        self.root = root

        # Generate/calculate commonly used values/properties
        self.bayer_map = self._generate_bayer_map()
        self.saturation = 2**self.bit_depth - 1
        self.bands = self.colour_description

    def load_settings(self):
        """
        Load a settings file
        """
        settings = load_json(self.root/"settings.json")

        # Convert the input exposures to floating point numbers
        settings["exposure_min"] = _convert_exposure_time(settings["exposure_min"])
        settings["exposure_max"] = _convert_exposure_time(settings["exposure_max"])

        # Add settings to the camera
        self.settings = self.Settings(**settings)

    def __repr__(self):
        """
        Text representation of the Camera object
        """
        if self.root is None:
            return f"{self.name} (not from file)"
        else:
            return f"{self.name} (from `{self.root}`)"

    def __str__(self):
        """
        Output for `print(Camera)`
        """
        combiner = "\n\t"
        device_name = f"{self.name}"
        manufacturer = f"manufacturer: {self.manufacturer}"
        internal_name = f"internal name: {self.name_internal}"
        if self.root is None:
            source = f"not from file"
        else:
            source = f"from `{self.root}`"
        calibration_list = f"calibration data: {self.check_calibration_data()}"

        text = combiner.join([device_name, manufacturer, internal_name, source, calibration_list])
        return text

    def _as_dict(self):
        """
        Generate a dictionary containing the Camera metadata, similar to the
        inputs to __init__.
        """
        dictionary = {prop: getattr(self, prop) for prop in self.property_list}
        return dictionary

    def _generate_bayer_map(self):
        """
        Generate a Bayer map, with the Bayer channel (RGBG2) for each pixel.
        """
        bayer_map = np.zeros(self.image_shape, dtype=int)
        bayer_map[0::2, 0::2] = self.bayer_pattern[0][0]
        bayer_map[0::2, 1::2] = self.bayer_pattern[0][1]
        bayer_map[1::2, 0::2] = self.bayer_pattern[1][0]
        bayer_map[1::2, 1::2] = self.bayer_pattern[1][1]
        return bayer_map

    def generate_bias_map(self):
        """
        Generate a Bayer-aware map of bias values from the camera information.
        """
        bayer_map = self._generate_bayer_map()
        for j, bias_value in enumerate(self.bias):
            bayer_map[bayer_map == j] = bias_value
        return bayer_map

    def _load_bias_map(self):
        """
        Load a bias map from the root folder or from the camera information.
        """
        # First try using a data-based bias map from file
        try:
            bias_map = bias_readnoise.load_bias_map(self.root)

        # If a data-based bias map does not exist or cannot be loaded, use camera information instead
        except (FileNotFoundError, OSError, TypeError):
            bias_map = self.generate_bias_map()
            self.bias_type = "Metadata"

        # If a data-based bias map was found, indicate this in the self.bias_type tag
        else:
            self.bias_type = "Measured"

        # Whatever bias map was used, save it to this object so it need not be re-loaded in the future
        finally:
            self.bias_map = bias_map

    def _load_readnoise_map(self):
        """
        Load a readnoise map for this sensor, from the root folder.
        """
        # Try to use a data-based read-noise map
        try:
            readnoise = bias_readnoise.load_readnoise_map(self.root)

        # If a data-based read-noise map does not exist or cannot be loaded, return an empty (None) object and warn
        except (FileNotFoundError, OSError, TypeError):
            readnoise = None
            print(f"Could not find a readnoise map in the folder `{self.root}`")

        # The read-noise map is saved to this object
        self.readnoise = readnoise

    def _load_dark_current_map(self):
        """
        Load a dark current map from the root folder - if none is available, return 0 everywhere.
        """
        # Try to use a dark current map from file
        try:
            dark_current = dark.load_dark_current_map(self.root)

        # If a dark current map does not exist, return an empty one, and warn the user
        except (FileNotFoundError, OSError, TypeError):
            dark_current = np.zeros(self.image_shape)
            print(f"Could not find a dark current map in the folder `{self.root}` - using all 0 instead")

        # Whatever bias map was used, save it to this object so it need not be re-loaded in the future
        self.dark_current = dark_current

    def _generate_ISO_range(self):
        """
        Generate an array from 0 to the max iso.
        """
        return np.arange(0, self.settings.ISO_max+1, 1)

    def _load_iso_normalisation(self):
        """
        Load an ISO normalisation look-up table from the root folder.
        If none is available, make an estimate from the camera's ISO range.
        """
        # Try to use a lookup table from file
        try:
            lookup_table = iso.load_iso_lookup_table(self.root)

        # If a lookup table cannot be found, assume a linear relation and warn the user
        except (FileNotFoundError, OSError, TypeError):
            iso_range = self._generate_ISO_range()
            normalisation = iso_range / self.settings.ISO_min
            lookup_table = np.stack([iso_range, normalisation])
            print(f"No ISO lookup table found for {self.name}. Using naive estimate (ISO / min ISO). This may not be accurate.")

        # Whatever method was used, save the lookup table so it need not be looked up again
        self.iso_lookup_table = lookup_table

    def _load_gain_map(self):
        """
        Load a gain map from the root folder.
        """
        # Try to use a gain map from file
        try:
            gain_map = gain.load_gain_map(self.root)

        # If a gain map cannot be found, do not use any, and warn the user
        except (FileNotFoundError, OSError, TypeError):
            gain_map = None
            print(f"No gain map found for {self.name}.")

        # If a gain map was found, save it to this object so it need not be looked up again
        # If no gain map was found, save the None object to warn the user
        self.gain_map = gain_map

    def _load_flatfield_correction(self):
        """
        Load a flatfield correction model from the root folder, and generate a correction map.
        """
        # Try to use a flatfield model from file
        try:
            correction_map = flat.load_flatfield_correction(self.root, shape=self.image_shape)

        # If a flatfield map cannot be found, do not use any, and warn the user
        except (FileNotFoundError, OSError, TypeError):
            correction_map = None
            print(f"No flatfield model found for {self.name}.")

        # If a flatfield map was found, save it to this object so it need not be looked up again
        # If no flatfield map was found, save the None object to warn the user
        self.flatfield_map = correction_map

    def _load_spectral_response(self):
        """
        Load spectral response curves from the root folder.
        """
        # Try to use SRFs from file
        try:
            spectral_response = spectral.load_spectral_response(self.root)

        # If SRFs cannot be found, do not use any, and warn the user
        except (FileNotFoundError, OSError, TypeError):
            spectral_response = None
            print(f"No spectral response functions found for {self.name}.")

        # If SRFs were found, save it to this object so it need not be looked up again
        # Else, save the None object to warn the user
        self.spectral_response = spectral_response

    def check_calibration_data(self):
        """
        Check what calibration data have been loaded so far.
        """
        data_available = [data_type for data_type in self.calibration_data_all if hasattr(self, data_type)]
        return data_available

    def correct_bias(self, *data, **kwargs):
        """
        Correct data for bias using this sensor's data.
        Bias data are loaded from the root folder or from the camera information.
        """
        # If a bias map has not been loaded yet, do so
        if not hasattr(self, "bias_map"):
            self._load_bias_map()

        # Apply the bias correction
        data_corrected = bias_readnoise.correct_bias_from_map(self.bias_map, *data, **kwargs)
        return data_corrected

    def correct_dark_current(self, exposure_time, *data, **kwargs):
        """
        Calibrate data for dark current using this sensor's data.
        Dark current data are loaded from the root folder or estimated 0 in all pixels,
        if no data were available.
        """
        # If a dark current map has not been loaded yet, do so
        if not hasattr(self, "dark_current"):
            self._load_dark_current_map()

        # Apply the dark current correction
        data_corrected = dark.correct_dark_current_from_map(self.dark_current, exposure_time, *data, **kwargs)
        return data_corrected

    def normalise_iso(self, iso_values, *data):
        """
        Normalise data for their ISO speed using this sensor's lookup table.
        The ISO lookup table is loaded from the root folder.
        """
        # If a lookup table has not been loaded yet, do so
        if not hasattr(self, "iso_lookup_table"):
            self._load_iso_normalisation()

        # Apply the ISO normalisation
        data_corrected = iso.normalise_iso_general(self.iso_lookup_table, iso_values, *data)
        return data_corrected

    def convert_to_photoelectrons(self, *data):
        """
        Convert data from ADU to photoelectrons using this sensor's gain data.
        The gain data are loaded from the root folder.
        """
        # If a gain map has not been loaded yet, do so
        if not hasattr(self, "gain_map"):
            self._load_gain_map()

        # Assert that a gain map was loaded
        assert self.gain_map is not None, "Gain map unavailable"

        # If a gain map was available, apply it
        data_converted = gain.convert_to_photoelectrons_from_map(self.gain_map, *data)
        return data_converted

    def correct_flatfield(self, *data, **kwargs):
        """
        Correct data for flatfield using this sensor's flatfield data.
        The flatfield data are loaded from the root folder.
        """
        # If a flatfield map has not been loaded yet, do so
        if not hasattr(self, "flatfield_map"):
            self._load_flatfield_correction()

        # Assert that a flatfield map was loaded
        assert self.flatfield_map is not None, "Flatfield map unavailable"

        # If a flatfield map was available, apply it
        data_corrected = flat.correct_flatfield_from_map(self.flatfield_map, *data, **kwargs)
        return data_corrected

    def correct_spectral_response(self, data_wavelengths, *data):
        """
        Correct data for the sensor's spectral response functions.
        The spectral response data are loaded from the root folder.
        """
        # If the SRFs have not been loaded yet, do so
        if not hasattr(self, "spectral_response"):
            self._load_spectral_response()

        # Assert that SRFs were loaded
        assert self.spectral_response is not None, "Spectral response functions unavailable"

        # If SRFs were available, correct for them
        data_normalised = spectral.correct_spectra(self.spectral_response, data_wavelengths, *data)
        return data_normalised

    def demosaick(self, *data, **kwargs):
        """
        Demosaick data using this camera's Bayer pattern.
        """
        RGBG_data = raw.demosaick(self.bayer_map, *data, color_desc=self.bands, **kwargs)
        return RGBG_data

    def plot_gauss_maps(self, data, **kwargs):
        """
        Plot Gaussian maps using analyse.plot_gauss_maps.
        Uses this camera's Bayer pattern.
        """
        analyse.plot_gauss_maps(data, self.bayer_map, **kwargs)

    def plot_histogram_RGB(self, data, **kwargs):
        """
        Plot an RGB histogram maps using analyse.plot_gauss_maps.
        Uses this camera's Bayer pattern.
        """
        analyse.plot_histogram_RGB(data, self.bayer_map, **kwargs)

    def write_to_file(self, path):
        """
        Write metadata to a file.
        """
        write_json(self._as_dict(), path)

    @classmethod
    def read_from_file(cls, path):
        """
        Read camera information from a JSON file.
        """
        root = find_root_folder(path)
        properties = load_json(path)
        return cls(**properties, root=root)


dummy_camera = Camera(name="Dummy", manufacturer="SPECTACLE", name_internal="dummy-123", image_shape=[1080, 1920], raw_extension=".dng", bias=[0,0,0,0], bayer_pattern=[[0,1],[2,3]], bit_depth=11, colour_description="RGBG", root=Path(__file__).parent)


def load_json(path):
    """
    Read a JSON file.
    """
    with open(path, "r") as file:
        try:
            # Load the JSON file
            dump = json.load(file)
        except json.JSONDecodeError:
            # If the JSON file could not be read, e.g. because it is empty, raise an error
            raise ValueError(f"Could not read JSON file `{path}`.")
    return dump


def write_json(data, save_to):
    """
    Write a JSON file containing `data` to a path `save_to`.
    """
    with open(save_to, "w") as file:
        json.dump(data, file, indent=4)


def load_camera(root, return_filename=False):
    """
    Read the camera information JSON located in the `root` folder.

    If the location of the metadata file itself is given for `root`,
    this is handled.

    If `return_filename` is True, also return the exact filename used.
    """
    root = Path(root)

    assert root.exists(), f"Cannot load metadata from `{root}` as this location does not exist."

    if root.is_file():
        # If a file is given instead of a folder, look for the folder first
        root_original = root
        root = find_root_folder(root)
        print(f"load_metadata was given a file (`{root_original}`) instead of a folder. Found a correct root folder to use instead: `{root}`")

    filename = root/"camera.json"
    metadata = Camera.read_from_file(filename)
    return return_with_filename(metadata, filename, return_filename)
