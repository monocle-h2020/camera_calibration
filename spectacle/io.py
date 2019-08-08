import rawpy
import exifread
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import json
from .iso import model_generator as iso_model_generator
from .flat import apply_vignette_radial
from .config import spectacle_folder, results_folder

def load_raw_file(filename):
    """
    Load a raw file using rawpy's `imread` function. Return the rawpy object.
    """
    img = rawpy.imread(str(filename))
    return img

def load_raw_image(filename):
    """
    Load a raw file using rawpy's `imread` function. Return only the image data.
    """
    img = load_raw_file(filename)
    return img.raw_image

def load_raw_colors(filename):
    """
    Load a raw file using rawpy's `imread` function. Return only the Bayer colour data.
    """
    img = load_raw_file(str(filename))
    return img.raw_colors

def load_raw_image_multi(folder, pattern="*.dng"):
    """
    Load many raw files simultaneously and put their image data in a single array.
    """

    # Find all files in `folder` matching the given pattern `pattern`
    files = list(folder.glob(pattern))

    # Load the first file to get the Bayer color information (`colors`)
    # and the shape of the images
    file0 = load_raw_file(files[0])
    colors = file0.raw_colors

    # Create an array to fit the image contained in each file
    arrs = np.empty((len(files), *file0.raw_image.shape), dtype=np.uint16)

    # Include the already loaded first image in the array
    arrs[0] = file0.raw_image

    # Include the image data from the other files in the array
    for j, file in enumerate(files[1:], 1):
        arrs[j] = load_raw_image(file)

    return arrs, colors

def load_jpg_image(filename):
    """
    Load a raw file using pyplot's `imread` function. Return only the image data.
    """
    img = plt.imread(filename)
    return img

def load_jpg_multi(folder, pattern="*.jp*g"):
    """
    Load many jpg files simultaneously and put their image data in a single array.
    """

    # Find all files in `folder` matching the given pattern `pattern`
    files = list(folder.glob(pattern))

    # Load the first file to get the shape of the images
    img0 = load_jpg_image(files[0])

    # Create an array to fit the image contained in each file
    arrs = np.empty((len(files), *img0.shape), dtype=np.uint8)

    # Include the already loaded first image in the array
    arrs[0] = img0

    # Include the image data from the other files in the array
    for j, file in enumerate(files[1:], 1):
        arrs[j] = load_jpg_image(file)

    return arrs

def load_exif(filename):
    with open(filename, "rb") as f:
        exif = exifread.process_file(f)
    return exif

def absolute_filename(file):
    return file.absolute()

def expected_array_size(folder, pattern):
    files = sorted(folder.glob(pattern))
    array = np.load(files[0])
    return np.array(array.shape)

def array_size_dng(folder):
    return expected_array_size(folder, pattern="*_mean.npy")

def load_npy(folder, pattern, retrieve_value=absolute_filename, selection=np.s_[:], **kwargs):
    files = sorted(folder.glob(pattern))
    stacked = np.stack([np.load(f)[selection] for f in files])
    values = np.array([retrieve_value(f, **kwargs) for f in files])
    return values, stacked

def split_path(path, split_on):
    split_split_on = path.stem.split(split_on)[1]
    split_underscore = split_split_on.split("_")[0]
    return split_underscore

def split_pol_angle(path):
    split_name = split_path(path, "pol")
    val = float(split_name.split("_")[0])
    return val

def split_exposure_time(path):
    without_letters = path.stem.strip("t_jmeansd")  # strip underscores, leading t, trailing "mean"/"stds"
    if "_" in without_letters:
        numerator, denominator = without_letters.split("_")
        time = float(numerator)/float(denominator)
    else:
        time = float(without_letters)
    return time

def split_iso(path):
    split_name = split_path(path, "iso")
    val = int(split_name.split("_")[0])
    return val

def load_means(folder, **kwargs):
    values, means = load_npy(folder, "*_mean.npy", **kwargs)
    return values, means

def load_jmeans(folder, **kwargs):
    values, means = load_npy(folder, "*_jmean.npy", **kwargs)
    return values, means

def load_stds(folder, **kwargs):
    values, stds = load_npy(folder, "*_stds.npy", **kwargs)
    return values, stds

def load_jstds(folder, **kwargs):
    values, stds = load_npy(folder, "*_jstds.npy", **kwargs)
    return values, stds

def load_colour(stacks):
    colours = np.load(stacks/"colour.npy")
    return colours

def load_angle(stacks):
    offset_angle = np.loadtxt(stacks/"linearity"/"default_angle.dat").ravel()[0]
    return offset_angle

def path_from_input(argv):
    if len(argv) == 2:
        return Path(argv[1])
    else:
        return [Path(a) for a in argv[1:]]

def folders(input_path):
    assert isinstance(input_path, Path), f"Input path '{input_path}' is not a pathlib Path object"
    assert spectacle_folder in input_path.parents, f"Input path '{input_path}' is not in the SPECTACLE data folder '{spectacle_folder}'"
    subfolder = input_path.relative_to(spectacle_folder).parts[0]
    phone_root = spectacle_folder / subfolder

    subfolder_names = ["", "images", "stacks", "products", "results"]
    subfolders = [phone_root/name for name in subfolder_names]
    return subfolders

def replace_word_in_path(path, old, new):
    split = list(path.parts)
    split[split.index(old)] = new
    combined = Path("/".join(split))
    return combined

def replace_suffix(path, new):
    return (path.parent / path.stem).with_suffix(".jpg")

def load_bias(products):
    bias_map = np.load(products/"bias.npy")
    return bias_map

def read_json(path):
    file = open(path)
    dump = json.load(file)
    return dump

def read_iso_lookup_table(products):
    table = np.load(products/"iso_lookup_table.npy")
    return table

def read_iso_model(products):
    as_array = np.loadtxt(products/"iso_model.dat", dtype=str)
    model_type = as_array[0,0]
    parameters = as_array[1].astype(np.float64)
    errors     = as_array[2].astype(np.float64)
    model = iso_model_generator[model_type](*parameters)
    return model

def read_flat_field_correction(products, shape):
    parameters, errors = np.load(products/"flat_parameters.npy")
    correction_map = apply_vignette_radial(shape, parameters)
    return correction_map

def read_gain_table(path):
    table = np.load(path)
    ISO = split_iso(path)
    return ISO, table

def read_spectral_responses(results):
    try:  # use monochromator data if available
        as_array = np.load(results/"spectral_response/monochromator_curve.npy")
    except FileNotFoundError:
        as_array = np.load(results/"spectral_response/curve.npy")
    wavelengths = as_array[0]
    RGBG2 = as_array[1:5]
    RGBG2_error = as_array[5:]
    return wavelengths, RGBG2, RGBG2_error

def read_spectral_bandwidths(products):
    bandwidths = np.loadtxt(products/"spectral_bandwidths.dat")
    return bandwidths
