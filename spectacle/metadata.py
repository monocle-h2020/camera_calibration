"""
Code relating to camera metadata, such as generating or reading metadata files.
"""

import numpy as np
import json
from collections import namedtuple
from pathlib import Path

from . import raw, analyse, bias_readnoise, dark, iso, gain
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
        if (parent/"metadata.json").exists():
            root = parent
            break
    # If no metadata file was found, raise an error
    else:
        raise OSError(f"None of the parents of the input `{input_path}` include a 'metadata.json' file.")

    return root


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
    Class that represents a camera, providing some metadata used in common
    calibration/analysis tasks. Some class methods for common tasks are also
    included, such as generating a Bayer map.
    """
    # Properties a Camera can have
    Device = namedtuple("Device", ["manufacturer", "name"])
    Image = namedtuple("Image", ["shape", "raw_extension", "bias", "bayer_pattern", "bit_depth"])
    Settings = namedtuple("Settings", ["ISO_min", "ISO_max", "exposure_min", "exposure_max"])

    def __init__(self, device_properties, image_properties, settings, root=None):
        """
        Generate a Camera object based on input dictionaries containing the
        keys defined in Device, Image, and Settings
        """
        # Convert the input exposures to floating point numbers
        settings["exposure_min"] = _convert_exposure_time(settings["exposure_min"])
        settings["exposure_max"] = _convert_exposure_time(settings["exposure_max"])

        # Create named tuples based on the input dictionaries
        self.device = self.Device(**device_properties)
        self.image = self.Image(**image_properties)
        self.settings = self.Settings(**settings)

        # Generate/calculate commonly used values/properties
        self.bayer_map = self._generate_bayer_map()
        self.saturation = 2**self.image.bit_depth - 1

        # Root folder
        self.root = root

    def __repr__(self):
        """
        Output for `print(Camera)`, currently simply the name of the device
        """
        device_name = f"{self.device.manufacturer} {self.device.name}"
        return device_name

    def _as_dict(self):
        """
        Generate a dictionary containing the Camera metadata, similar to the
        inputs to __init__.
        """
        dictionary = {"device": self.device._asdict(),
                      "image": self.image._asdict(),
                      "settings": self.settings._asdict()}
        return dictionary

    def _generate_bayer_map(self):
        """
        Generate a Bayer map, with the Bayer channel (RGBG2) for each pixel.
        """
        bayer_map = np.zeros(self.image.shape, dtype=int)
        bayer_map[0::2, 0::2] = self.image.bayer_pattern[0][0]
        bayer_map[0::2, 1::2] = self.image.bayer_pattern[0][1]
        bayer_map[1::2, 0::2] = self.image.bayer_pattern[1][0]
        bayer_map[1::2, 1::2] = self.image.bayer_pattern[1][1]
        return bayer_map

    def generate_bias_map(self):
        """
        Generate a Bayer-aware map of bias values from the camera metadata
        """
        bayer_map = self._generate_bayer_map()
        for j, bias_value in enumerate(self.image.bias):
            bayer_map[bayer_map == j] = bias_value
        return bayer_map

    def _load_bias_map(self):
        """
        Load a bias map from file or from metadata
        """
        # First try using a data-based bias map from file
        try:
            bias_map = bias_readnoise.load_bias_map(self.root)

        # If a data-based bias map does not exist or cannot be loaded, use metadata instead
        except (FileNotFoundError, OSError):
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
        Load a readnoise map for this sensor
        """
        # Try to use a data-based read-noise map
        try:
            readnoise = bias_readnoise.load_readnoise_map(self.root)

        # If a data-based read-noise map does not exist or cannot be loaded, return an empty (None) object and warn
        except (FileNotFoundError, OSError):
            readnoise = None
            print(f"Could not find a readnoise map in the folder `{self.root}`")

        # The read-noise map is saved to this object
        self.readnoise = readnoise

    def _load_dark_current_map(self):
        """
        Load a dark current map from file - if none is available, return 0s
        """
        # Try to use a dark current map from file
        try:
            dark_current = dark.load_dark_current_map(self.root)

        # If a dark current map does not exist, return an empty one, and warn the user
        except (FileNotFoundError, OSError):
            dark_current = np.zeros(self.image.shape)
            print(f"Could not find a dark current map in the folder `{self.root}` - using all 0 instead")

        # Whatever bias map was used, save it to this object so it need not be re-loaded in the future
        self.dark_current = dark_current

    def _generate_ISO_range(self):
        """
        Generate an array from 0 to the max iso
        """
        return np.arange(0, self.settings.ISO_max+1, 1)

    def _load_iso_normalisation(self):
        """
        Load an ISO normalisation look-up table from file or from parameters
        """
        # Try to use a lookup table from file
        try:
            lookup_table = iso.load_iso_lookup_table(self.root)

        # If a lookup table cannot be found, assume a linear relation and warn the user
        except (FileNotFoundError, OSError):
            iso_range = self._generate_ISO_range()
            normalisation = iso_range / self.settings.ISO_min
            lookup_table = np.stack([iso_range, normalisation])
            print(f"No ISO lookup table found for {self.device.name}. Using naive estimate (ISO / min ISO). This may not be accurate.")

        # Whatever method was used, save the lookup table so it need not be looked up again
        self.iso_lookup_table = lookup_table

    def _load_gain_map(self):
        """
        Load a gain map from file
        """
        # Try to use a gain map from file
        try:
            gain_map = gain.load_gain_map(self.root)

        # If a gain map cannot be found, do not use any, and warn the user
        except (FileNotFoundError, OSError):
            gain_map = None
            print(f"No gain map found for {self.device.name}.")

        # If a gain map was found, save it to this object so it need not be looked up again
        # If no gain map was found, save the None object to warn the user
        self.gain_map = gain_map

    def correct_bias(self, *data, **kwargs):
        """
        Correct data for bias using this sensor's bias data
        """
        # If a bias map has not been loaded yet, do so
        if not hasattr(self, "bias_map"):
            self._load_bias_map()

        # Apply the bias correction
        data_corrected = bias_readnoise.correct_bias_from_map(self.bias_map, *data, **kwargs)
        return data_corrected

    def correct_dark_current(self, exposure_time, *data, **kwargs):
        """
        Calibrate data for dark current using this sensor's data
        """
        # If a dark current map has not been loaded yet, do so
        if not hasattr(self, "dark_current"):
            self._load_dark_current_map()

        # Apply the dark current correction
        data_corrected = dark.correct_dark_current_from_map(self.dark_current, exposure_time, *data, **kwargs)
        return data_corrected

    def normalise_iso(self, iso_values, *data):
        """
        Normalise data for their ISO speed using this sensor's lookup table
        """
        # If a lookup table has not been loaded yet, do so
        if not hasattr(self, "iso_lookup_table"):
            self._load_iso_normalisation()

        # Apply the ISO normalisation
        data_corrected = iso.normalise_iso_general(self.iso_lookup_table, iso_values, *data)
        return data_corrected

    def convert_to_photoelectrons(self, *data):
        """
        Convert data from ADU to photoelectrons using this sensor's gain data
        """
        # If a gain map has not been loaded yet, do so
        if not hasattr(self, "gain_map"):
            self._load_gain_map()

        # Assert that a gain map was loaded
        assert self.gain_map is not None, "Gain map unavailable"

        # If a gain map was available, apply it
        data_converted = gain.convert_to_photoelectrons_from_map(self.gain_map, *data)
        return data_converted

    def write_to_file(self, path):
        """
        Write metadata to a file.
        """
        write_json(self._as_dict(), path)

    def demosaick(self, *data, **kwargs):
        """
        Demosaick data using this camera's Bayer pattern.
        """
        RGBG_data = raw.demosaick(self.bayer_map, *data, **kwargs)
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

    @classmethod
    def read_from_file(cls, path):
        """
        Read metadata from a JSON file.
        """
        root = find_root_folder(path)
        full_dictionary = load_json(path)
        device_properties, image_properties, settings = full_dictionary.values()
        return cls(device_properties, image_properties, settings, root=root)


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
        json.dump(data, file)


def load_metadata(root, return_filename=False):
    """
    Read the metadata JSON located in the `root` folder.

    If `return_filename` is True, also return the exact filename used.
    """
    filename = root/"metadata.json"
    metadata = Camera.read_from_file(filename)
    return return_with_filename(metadata, filename, return_filename)
