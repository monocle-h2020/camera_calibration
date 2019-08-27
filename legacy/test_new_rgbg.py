import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import raw, plot, io
from spectacle.general import gaussMd

def gauss_nan(data, *args, **kwargs):
    V = data.copy()
    V[np.isnan(data)] = 0
    VV = gaussMd(V, *args, **kwargs)

    W = np.ones_like(data)
    W[np.isnan(data)] = 0
    WW = gaussMd(W, *args, **kwargs)

    return VV/WW

def pull_apart2(raw_img, color_pattern, color_desc="RGBG", remove=True):
    unique_colours = np.unique(color_pattern)

    stack = np.tile(np.nan, (*unique_colours.shape, *raw_img.shape))
    for j in range(len(unique_colours)):
        indices = np.where(color_pattern == j)
        stack[j][indices] = raw_img[indices]

    if remove and len(set(color_desc)) != len(color_desc):
        to_remove = []
        for j in range(1, len(unique_colours)):
            colour = color_desc[j]
            previous = color_desc[:j]
            try:
                ind_previous = previous.index(colour)
            except ValueError:
                continue
            stack[ind_previous][color_pattern == j] = stack[j][color_pattern == j]
            to_remove.append(j)
        clean_stack = np.delete(stack, to_remove, axis=0)
    else:
        clean_stack = stack.copy()

    assert len(np.where(np.nansum(clean_stack, axis=0) != raw_img)[0]) == 0

    return clean_stack

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
results_readnoise = results/"readnoise"

isos, stds  = io.load_stds  (folder, retrieve_value=io.split_iso)
colours     = io.load_colour(stacks)

s = stds[isos.argmin()]

RGBG, _ = raw.pull_apart(s, colours)

unique_colours = np.unique(colours).shape[0]

RGB = pull_apart2(s, colours)
