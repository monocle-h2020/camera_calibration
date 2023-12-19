"""
Code relating to Camera objects, which store camera information such the name,
manufacturer, and RGBG pattern of the camera.
The Camera object also provides an easy interface for data calibration/correction.
"""

import json
from collections import namedtuple
from os import makedirs
from pathlib import Path

import numpy as np

from . import analyse, bias_readnoise, dark, flat, gain, iso, raw, spectral
from .general import find_matching_file, return_with_filename

# Empty slice that just selects all data - used as default argument
all_data = np.s_[:]


def find_root_folder(input_path):
    """
    For a given `input_path`, find the root folder, containing the standard
    sub-folders (calibration, analysis, stacks, etc.)
    """
    input_path = Path(input_path)

    # Loop through the input_path's parents until a metadata JSON file is found
    for parent in [input_path, *input_path.parents]:
        # If a metadata file is found, use the containing folder as the root folder
        try:
            json_file = find_matching_file(parent, "data.json")
        except FileNotFoundError:
            continue
        else:
            root = parent
            break
    # If no metadata file was found, raise an error
    else:
        raise OSError(f"None of the parents of the input `{input_path}` include a camera data JSON file.")

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


def makedirs_without_file(path):
    """
    Similar to os.makedirs, creating a folder tree, but checks if
    the given path is file-like (has a file extension). If yes, the
    final element of the tree is assumed to be a file and ignored.
    """
    # If the path has no suffix, create it as well as its parents
    if path.suffix == "":
        path_for_makedirs = path
    # If it does, only create the path's parents
    else:
        path_for_makedirs = path.parent
    makedirs(path_for_makedirs, exist_ok=True)


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
    _Settings = namedtuple("Settings", ["ISO_min", "ISO_max", "exposure_min", "exposure_max"])

    calibration_data_all = ["settings", "bias_map", "readnoise", "dark_current", "iso_lookup_table", "gain_map", "flatfield_map", "spectral_response", "spectral_bands", "XYZ_matrix"]

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

        # Load settings if available
        try:
            self.load_settings()
        except FileNotFoundError:
            pass

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

    def central_slice(self, width_x, width_y):
        """
        Generate a numpy slice object around the center of an image, with widths
        width_x, width_y. Note that there may be rounding errors for odd widths.
        """
        dx, dy = width_x//2, width_y//2
        midx, midy = np.array(self.image_shape)//2
        center = np.s_[..., midx-dx:midx+dx, midy-dy:midy+dy]
        return center

    def load_settings(self):
        """
        Load a settings file
        """
        filename = find_matching_file(self.root/"calibration", "settings.json")
        settings = load_json(filename)

        # Convert the input exposures to floating point numbers
        settings["exposure_min"] = _convert_exposure_time(settings["exposure_min"])
        settings["exposure_max"] = _convert_exposure_time(settings["exposure_max"])

        # Add settings to the camera
        self.settings = self._Settings(**settings)

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
            try:
                iso_range = self._generate_ISO_range()
            except AttributeError:
                iso_range = np.arange(0, 2000, 1)
                print("No Settings file loaded, so did not know native ISO range. Using 0-2000 instead.")
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

    def load_spectral_bands(self):
        """
        Load spectral bands from the root folder.
        """
        # Try to load from file
        try:
            spectral_bands = spectral.load_spectral_bands(self.root)

        # If spectral bands cannot be found, do not use any, and warn the user
        except (FileNotFoundError, OSError, TypeError):
            spectral_bands = None
            print(f"No spectral band data found for {self.name}.")

        # If spectral bands were found, save it to this object so it need not be looked up again
        # Else, save the None object to warn the user
        self.spectral_bands = spectral_bands

    def _load_XYZ_matrix(self):
        """
        Load the RGB -> XYZ conversion matrix from the root folder.
        """
        # Try to load from file
        try:
            XYZ_matrix = spectral.load_XYZ_matrix(self.root)

        # If a matrix cannot be found, do not use one, and warn the user
        except (FileNotFoundError, OSError, TypeError):
            XYZ_matrix = None
            print(f"No RGB->XYZ matrix found for {self.name}.")

        # If an XYZ matrix was found, save it to this object so it need not be looked up again
        # Else, save the None object to warn the user
        self.XYZ_matrix = XYZ_matrix

    def load_all_calibrations(self):
        """
        Load all available calibration data for this camera.
        """
        for func in [self.load_settings, self._load_bias_map, self._load_dark_current_map, self._load_flatfield_correction, self._load_gain_map, self._load_iso_normalisation, self._load_readnoise_map, self._load_spectral_response, self.load_spectral_bands, self._load_XYZ_matrix]:
            func()

    def check_calibration_data(self):
        """
        Check what calibration data have been loaded so far.
        """
        data_available = [data_type for data_type in self.calibration_data_all if hasattr(self, data_type)]
        data_available = [data_type for data_type in data_available if getattr(self, data_type) is not None]
        return data_available

    def correct_bias(self, data, selection=all_data):
        """
        Correct data for bias using this sensor's data.
        Bias data are loaded from the root folder or from the camera information.
        """
        # If a bias map has not been loaded yet, do so
        if not hasattr(self, "bias_map"):
            self._load_bias_map()

        # Select the relevant data
        bias_map = self.bias_map[selection]

        # Apply the bias correction
        data_corrected = bias_readnoise.correct_bias_from_map(bias_map, data)
        return data_corrected

    def correct_dark_current(self, exposure_time, data, selection=all_data):
        """
        Calibrate data for dark current using this sensor's data.
        Dark current data are loaded from the root folder or estimated 0 in all pixels,
        if no data were available.
        """
        # If a dark current map has not been loaded yet, do so
        if not hasattr(self, "dark_current"):
            self._load_dark_current_map()

        # Select the relevant data
        dark_current = self.dark_current[selection]

        # Apply the dark current correction
        data_corrected = dark.correct_dark_current_from_map(dark_current, exposure_time, data)
        return data_corrected

    def normalise_iso(self, iso_values, data):
        """
        Normalise data for their ISO speed using this sensor's lookup table.
        The ISO lookup table is loaded from the root folder.
        """
        # If a lookup table has not been loaded yet, do so
        if not hasattr(self, "iso_lookup_table"):
            self._load_iso_normalisation()

        # Apply the ISO normalisation
        data_corrected = iso.normalise_iso(self.iso_lookup_table, iso_values, data)
        return data_corrected

    def convert_to_photoelectrons(self, data, selection=all_data):
        """
        Convert data from ADU to photoelectrons using this sensor's gain data.
        The gain data are loaded from the root folder.
        """
        # If a gain map has not been loaded yet, do so
        if not hasattr(self, "gain_map"):
            self._load_gain_map()

        # Assert that a gain map was loaded
        assert self.gain_map is not None, "Gain map unavailable"

        # Select the relevant data
        gain_map = self.gain_map[selection]

        # If a gain map was available, apply it
        data_converted = gain.convert_to_photoelectrons_from_map(gain_map, data)
        return data_converted

    def correct_flatfield(self, data, selection=all_data, **kwargs):
        """
        Correct data for flatfield using this sensor's flatfield data.
        The flatfield data are loaded from the root folder.
        """
        # If a flatfield map has not been loaded yet, do so
        if not hasattr(self, "flatfield_map"):
            self._load_flatfield_correction()

        # Assert that a flatfield map was loaded
        assert self.flatfield_map is not None, "Flatfield map unavailable"

        # Select the relevant data
        flatfield_map = self.flatfield_map[selection]

        # If a flatfield map was available, apply it
        data_corrected = flat.correct_flatfield_from_map(flatfield_map, data, **kwargs)
        return data_corrected

    def correct_spectral_response(self, data_wavelengths, data, **kwargs):
        """
        Correct data for the sensor's spectral response functions.
        The spectral response data are loaded from the root folder.
        """
        # If the SRFs have not been loaded yet, do so
        if not hasattr(self, "spectral_response"):
            self._load_spectral_response()

        # Assert that SRFs were loaded
        assert self.spectral_response is not None, "Spectral response functions unavailable"

        # Get the wavelengths and SRFs from the data
        wavelengths = self.spectral_response[0]
        SRFs = self.spectral_response[1:5]

        # If SRFs were available, correct for them
        data_normalised = spectral.correct_spectra(wavelengths, SRFs, data_wavelengths, data, **kwargs)
        return data_normalised

    def convolve(self, data_wavelengths, data):
        """
        Spectral convolution of a data set (`data_wavelengths`, `data_response`) over
        the camera's spectral bands.
        """
        # If the SRFs have not been loaded yet, do so
        if not hasattr(self, "spectral_response"):
            self._load_spectral_response()

        # Assert that SRFs were loaded
        assert self.spectral_response is not None, "Spectral response functions unavailable"

        # If SRFs were available, apply spectral convolution
        data_convolved = np.array([spectral.convolve(self.spectral_response[0], SRF, data_wavelengths, data) for SRF in self.spectral_response[1:5]])  # Loop over the RGBG2 spectral response functions
        return data_convolved

    def convolve_multi(self, data_wavelengths, data):
        """
        Spectral convolution of a data set (`data_wavelengths`, `data_response`) over
        the camera's spectral bands.

        Loops over multiple spectra at once.
        """
        # If the SRFs have not been loaded yet, do so
        if not hasattr(self, "spectral_response"):
            self._load_spectral_response()

        # Assert that SRFs were loaded
        assert self.spectral_response is not None, "Spectral response functions unavailable"

        # If SRFs were available, apply spectral convolution
        data_convolved = np.array([spectral.convolve_multi(self.spectral_response[0], SRF, data_wavelengths, data) for SRF in self.spectral_response[1:5]])  # Loop over the RGBG2 spectral response functions
        return data_convolved

    # Convert RGBG2 to RGB data - does not require any camera-specific data,
    # but useful to have as a class method
    convert_RGBG2_to_RGB = staticmethod(spectral.convert_RGBG2_to_RGB)

    def convert_to_XYZ(self, data, axis=None):
        """
        Convert RGB or RGBG2 data to XYZ using the sensor's conversion matrix.
        The conversion matrix is loaded from the root folder.
        `axis` is the RGB axis. If None is provided, one is automatically looked for.
        """
        # If the XYZ conversion matrix has not been loaded yet, do so
        if not hasattr(self, "XYZ_matrix"):
            self._load_XYZ_matrix()

        # Assert that a conversion matrix was loaded
        assert self.XYZ_matrix is not None, "RGB to XYZ conversion matrix unavailable"

        # If a conversion matrix was available, use it in the conversion
        data_XYZ = spectral.convert_to_XYZ(self.XYZ_matrix, data, axis=axis)
        return data_XYZ

    def colour_space(self):
        """
        Calculate the base vectors in xy chromaticity space for this camera's colour space.
        """
        # If the XYZ matrix has not been loaded yet, do so
        if not hasattr(self, "XYZ_matrix"):
            self._load_XYZ_matrix()

        # Assert that XYZ matrix was loaded
        assert self.XYZ_matrix is not None, "RGB -> XYZ conversion matrix unavailable"

        # If the XYZ matrix was available, convert its base vectors to xy
        colour_space = spectral.calculate_xy_base_vectors(self.XYZ_matrix)

        return colour_space

    def demosaick(self, data, selection=all_data, **kwargs):
        """
        Demosaick data using this camera's Bayer pattern.
        """
        # Select the relevant data
        bayer_map = self.bayer_map[selection]

        # Demosaick the data
        RGBG_data = raw.demosaick(bayer_map, data, color_desc=self.bands, **kwargs)
        return RGBG_data

    def plot_spectral_response(self, **kwargs):
        """
        Plot the camera's spectral response function.
        """
        # If the SRFs have not been loaded yet, do so
        if not hasattr(self, "spectral_response"):
            self._load_spectral_response()

        spectral.plot_spectral_responses([self.spectral_response[0]], [self.spectral_response[1:5]], labels=[self.name], **kwargs)

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

    def filename_analysis(self, suffix, makefolders=False):
        """
        Shortcut to get a filename in the `analysis` folder with a given `suffix`.
        """
        filename = self.root/"analysis"/suffix

        # Make sure the relevant folders exist
        if makefolders:
            makedirs_without_file(filename)

        return filename

    def filename_intermediaries(self, suffix, makefolders=False):
        """
        Shortcut to get a filename in the `intermediaries` folder with a given `suffix`.
        """
        filename = self.root/"intermediaries"/suffix

        # Make sure the relevant folders exist
        if makefolders:
            makedirs_without_file(filename)

        return filename

    def filename_calibration(self, suffix, makefolders=True):
        """
        Shortcut to get a filename in the `calibration` folder with a given `suffix`.

        This filename will include the camera name.
        """
        filename = self.root/f"calibration/{self.name_underscore}_{suffix}"

        # Make sure the relevant folders exist
        if makefolders:
            makedirs_without_file(filename)

        return filename

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

    filename = find_matching_file(root, "data.json")
    metadata = Camera.read_from_file(filename)
    return return_with_filename(metadata, filename, return_filename)
