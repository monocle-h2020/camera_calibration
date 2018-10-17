import rawpy
import exifread
import numpy as np
from pathlib import Path
from matplotlib import pyplot as plt
import json

def load_dng_raw(filename):
    img = rawpy.imread(str(filename))
    return img

def load_colors(filename):
    img = load_dng_raw(str(filename))
    return img.raw_colors

def load_dng_many(folder, pattern="*.dng"):
    files = list(folder.glob(pattern))
    file0 = load_dng_raw(files[0])
    colors = file0.raw_colors
    arrs = np.empty((len(files), *file0.raw_image.shape), dtype=np.uint16)
    arrs[0] = file0.raw_image
    for j, file in enumerate(files[1:], 1):
        arrs[j] = load_dng_raw(file).raw_image

    return arrs, colors

def load_jpg(filename):
    img = plt.imread(filename)
    return img

def load_jpg_many(folder, pattern="*.jp*g"):
    files = list(folder.glob(pattern))
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

def absolute_filename(file):
    return file.absolute

def load_npy(folder, pattern, retrieve_value=absolute_filename, **kwargs):
    files = sorted(folder.glob(pattern))
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

def path_from_input(argv):
    if len(argv) == 2:
        return Path(argv[1])
    else:
        return [Path(a) for a in argv[1:]]

def folders(data_folder):
    assert "data" in data_folder.parts
    phone_root = Path("/".join(data_folder.parts[:2]))

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

def read_gain_lookup_table(results):
    table = np.load(results/"gain/gain_lookup_table.npy")
    return table
