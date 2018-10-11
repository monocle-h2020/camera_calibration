import rawpy
import exifread
import glob
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt

def load_dng_raw(filename):
    img = rawpy.imread(filename)
    return img

def load_colors(filename):
    img = load_dng_raw(filename)
    return img.raw_colors

def load_dng_many(pattern, return_colors=False):
    files = glob.glob(pattern)
    file0 = load_dng_raw(files[0])
    colors = file0.raw_colors
    arrs = np.empty((len(files), *file0.raw_image.shape), dtype=np.uint16)
    arrs[0] = file0.raw_image
    for j, file in enumerate(files[1:], 1):
        arrs[j] = load_dng_raw(file).raw_image

    if return_colors:
        return arrs, colors
    else:
        return arrs

def load_jpg(filename):
    img = plt.imread(filename)
    return img

def load_jpg_many(pattern):
    files = glob.glob(pattern)
    img0 = load_jpg(files[0])
    arrs = np.empty((len(files), *img0.shape), dtype=np.uint8)
    arrs[0] = img0
    for j, file in enumerate(files[1:], 1):
        arrs[j] = load_jpg(file)
    return arrs

def load_exif(filename):
    with open(filename, "rb") as f:
        exif = exifread.process_file(f)
    return exif

def load_bias_ron(iso, folder="results/bias/"):
    bias = np.load(f"{folder}/bias_mean_iso{iso}.npy")
    ron  = np.load(f"{folder}/bias_stds_iso{iso}.npy")
    return bias, ron

def load_arrs(prefix):
    mean = np.load(f"{prefix}_mean.npy")
    stds = np.load(f"{prefix}_stds.npy")
    colr = np.load(f"{prefix}_colr.npy")
    return mean, stds, colr

def load_jarrs(prefix):
    jmean = np.load(f"{prefix}_jmean.npy")
    jstds = np.load(f"{prefix}_jstds.npy")
    return jmean, jstds

def absolute_filename(file):
    return file.absolute

def load_npy(folder, pattern, retrieve_value=absolute_filename, **kwargs):
    files = list(folder.glob(pattern))
    stacked = np.stack([np.load(f) for f in files])
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

def split_iso(path):
    split_name = split_path(path, "iso")
    val = int(split_name.split("_")[0])
    return val

def load_means(folder, **kwargs):
    values, means = load_npy(folder, "*_mean.npy", **kwargs)
    return values, means

def load_stds(folder, **kwargs):
    values, stds = load_npy(folder, "*_stds.npy", **kwargs)
    return values, stds

def load_colour(folder):
    # change to just use stacks folder as input
    folder_main  = Path("/".join(folder.parts[:3]))
    colours = np.load(folder_main/"colour.npy")
    return colours

def path_from_input(argv):
    return Path(argv[1])

def folders(data_folder):
    assert "data" in data_folder.parts
    phone_root = Path("/".join(data_folder.parts[:2]))

    subfolder_names = ["", "images", "stacks", "products", "results"]
    subfolders = [phone_root/name for name in subfolder_names]
    return subfolders
