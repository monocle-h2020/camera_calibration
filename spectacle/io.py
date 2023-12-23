import os
from pathlib import Path
from string import ascii_letters
from typing import Callable

import exifread
import numpy as np
import rawpy
from matplotlib import pyplot as plt
from tqdm import tqdm

from .camera import find_root_folder, load_camera, load_json, write_json
from .general import find_matching_file

# Default save folder for results
results_folder = Path.home() / "SPECTACLE_results"
if not results_folder.exists():
    os.makedirs(results_folder)
    print(f"Created SPECTACLE results folder: {results_folder}")

def path_from_input(argv):
    """
    Turn command-line input(s) into Path objects.
    """
    if len(argv) == 2:
        return Path(argv[1])
    else:
        return [Path(a) for a in argv[1:]]


def load_raw_file(filename):
    """
    Load a raw file using rawpy's `imread` function. Return the rawpy object.
    """
    # Convert filename to str because rawpy does not support Path
    filename_as_str = str(filename)
    img = rawpy.imread(filename_as_str)
    return img


def load_raw_image(filename):
    """
    Load a raw file using rawpy's `imread` function. Return only the image
    data.
    """
    img = load_raw_file(filename)
    return img.raw_image


def load_raw_colors(filename):
    """
    Load a raw file using rawpy's `imread` function. Return only the Bayer
    colour data.
    """
    img = load_raw_file(filename)
    return img.raw_colors


def load_raw_image_postprocessed(filename, **kwargs):
    """
    Load a raw file using rawpy's `imread` function and post-process it.
    Return the post-processed image data.
    """
    img = load_raw_file(filename)
    img_post = img.postprocess(**kwargs)
    return img_post


def load_raw_image_multi(folder: Path | str, pattern="*.dng", progressbar=True, leave_progressbar=False) -> np.ndarray[np.uint16]:
    """
    Load many raw files simultaneously and put their image data in a single array.
    Shows a progressbar if desired (default: True).
    """
    # Ensure the folder is a Path
    folder = Path(folder)

    # Find all files in `folder` matching the given pattern `pattern`
    files = list(folder.glob(pattern))
    nfiles = len(files)

    # Load the first file to get the shape of the images
    file0 = load_raw_file(files[0])

    # Create an array to fit the image contained in each file
    arrs = np.empty((nfiles, *file0.raw_image.shape), dtype=np.uint16)

    # Include the already loaded first image in the array
    arrs[0] = file0.raw_image

    # Include the image data from the other files in the array
    for j, file in tqdm(enumerate(files[1:], 1), desc="Loading RAW images", unit="file", total=nfiles, initial=1, disable=not progressbar, leave=leave_progressbar):
        arrs[j] = load_raw_image(file)

    return arrs


def load_jpg_image(filename):
    """
    Load a raw file using pyplot's `imread` function.
    Return only the image data.
    """
    img = plt.imread(filename)
    return img


def load_jpg_multi(folder: Path | str, pattern="*.jp*g", progressbar=True, leave_progressbar=False) -> np.ndarray[np.uint8]:
    """
    Load many jpg files simultaneously and put their image data in a single array.
    Shows a progressbar if desired (default: True).
    """
    # Ensure the folder is a Path
    folder = Path(folder)

    # Find all files in `folder` matching the given pattern `pattern`
    files = list(folder.glob(pattern))
    nfiles = len(files)

    # Load the first file to get the shape of the images
    img0 = load_jpg_image(files[0])

    # Create an array to fit the image contained in each file
    arrs = np.empty((nfiles, *img0.shape), dtype=np.uint8)

    # Include the already loaded first image in the array
    arrs[0] = img0

    # Include the image data from the other files in the array
    for j, file in tqdm(enumerate(files[1:], 1), desc="Loading JPEG images", unit="file", total=nfiles, initial=1, disable=not progressbar, leave=leave_progressbar):
        arrs[j] = load_jpg_image(file)

    return arrs


def load_exif(filename):
    """
    Load the EXIF data in an image using exifread's `process_file` function.
    Return all EXIF data.
    """
    with open(filename, "rb") as f:
        exif = exifread.process_file(f)
    return exif


def absolute_filename(file):
    """
    Return the absolute filename of a given Path object `file`.
    This mostly serves as a dummy example function for load_npy.
    """
    return file.absolute()


def expected_array_size(folder, pattern):
    """
    Find the required array size when loading files from `folder` that follow
    the pattern `pattern`, e.g. all .DNG files.
    """
    # Make sure `folder` is a Path-like object
    folder = Path(folder)
    files = sorted(folder.glob(pattern))
    array = np.load(files[0])
    return np.array(array.shape)


def load_npy(folder: Path | str, pattern: str, retrieve_value: Callable=absolute_filename, selection: slice=np.s_[:], progressbar=True, leave_progressbar=False, **kwargs) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a series of .npy (NumPy binary) files from `folder` following a pattern `pattern`.
    Returns the contents of the .npy files as well as a list of values based on their parsing their filenames with a function given in the `retrieve_value` keyword.
    Any **kwargs are passed to `retrieve_value`.
    Only return array elements included in `selection` (default: all).
    """
    # Make sure `folder` is a Path-like object
    folder = Path(folder)

    # Load the data
    filenames = tqdm(sorted(folder.glob(pattern)), desc="Loading .npy files", unit="file", disable=not progressbar, leave=leave_progressbar)

    stacked, values = zip(*[(np.load(f)[selection], retrieve_value(f, **kwargs)) for f in filenames])
    stacked = np.stack(stacked)
    values = np.array(values)

    return values, stacked


def split_path(path, split_on):
    """
    Split a pathlib Path object `path` on a string `split_on`.

    Input `path` is converted to a Path-type object, but
    output `split_underscore` is (and must be) a str object
    """
    # Make sure `path` is a Path-type object
    path = Path(path)
    split_split_on = path.stem.split(split_on)[1]
    split_underscore = split_split_on.split("_")[0]
    return split_underscore


def split_pol_angle(path):
    """
    Retrieve a polariser angle from a path `path`. Expects the path to contain
    a string `polX` with X the polariser angle.
    """
    split_name = split_path(path, "pol")
    val = float(split_name.split("_")[0])
    return val


def split_exposure_time(path):
    """
    Retrieve an exposure time from a path `path`. Handles inverted exposure
    times, e.g. 1/1000 seconds.
    For a filename `x.z`, the returned value is `float(x)`.
    For a filename `x_y.z`, the returned value is `float(x/y)`.
    """
    without_letters = path.stem.strip(ascii_letters+"_")  # strip underscores, leading t, trailing "mean"/"stds"
    if "_" in without_letters:
        numerator, denominator = without_letters.split("_")
        time = float(numerator)/float(denominator)
    else:
        time = float(without_letters)
    return time


def split_iso(path):
    """
    Retrieve an ISO speed from a path `path`. Expects the path to contain
    a string `isoX` with X the ISO speed.
    """
    split_name = split_path(path, "iso")
    val = int(split_name.split("_")[0])
    return val


def load_means(folder, **kwargs):
    """
    Quickly load all the mean RAW image stacks in a given folder.

    Load the files in `folder` that follow the pattern `*_mean.npy`.
    Any additional **kwargs are passed to `load_npy`.
    """
    values, means = load_npy(folder, "*_mean.npy", **kwargs)
    return values, means


def load_jmeans(folder, **kwargs):
    """
    Quickly load all the mean JPG image stacks in a given folder.

    Load the files in `folder` that follow the pattern `*_jmean.npy`.
    Any additional **kwargs are passed to `load_npy`.
    """
    values, means = load_npy(folder, "*_jmean.npy", **kwargs)
    return values, means


def load_stds(folder, **kwargs):
    """
    Quickly load all the standard deviation RAW image stacks in a given folder.

    Load the files in `folder` that follow the pattern `*_stds.npy`.
    Any additional **kwargs are passed to `load_npy`.
    """
    values, stds = load_npy(folder, "*_stds.npy", **kwargs)
    return values, stds


def load_jstds(folder, **kwargs):
    """
    Quickly load all the standard deviation JPG image stacks in a given folder.

    Load the files in `folder` that follow the pattern `*_jstds.npy`.
    Any additional **kwargs are passed to `load_npy`.
    """
    values, stds = load_npy(folder, "*_jstds.npy", **kwargs)
    return values, stds


def replace_word_in_path(path, old, new):
    """
    Replace the string `old` with the string `new` in a given `path`.
    """
    # Make sure `path` is a Path-type object
    path = Path(path)
    split = list(path.parts)
    split[split.index(old)] = new
    combined = Path("/".join(split))
    return combined


def replace_suffix(path, new_suffix):
    """
    Replace a suffix in a path with `new_suffix`
    """
    # Make sure `path` is a Path-type object
    path = Path(path)
    return (path.parent / path.stem).with_suffix(new_suffix)


def read_gain_table(path):
    table = np.load(path)
    ISO = split_iso(path)
    return ISO, table


def find_subfolders(path):
    """
    Find the subfolders of a given `path`
    """
    current_path, subfolders, files = next(os.walk(path))
    subfolders_as_paths = [path/subfolder for subfolder in subfolders]
    return subfolders_as_paths
