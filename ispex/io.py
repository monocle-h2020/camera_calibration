import rawpy
import exifread
import glob
import numpy as np

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

def load_exif(filename):
    with open(filename, "rb") as f:
        exif = exifread.process_file(f)
    return exif