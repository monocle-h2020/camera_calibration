import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import raw, plot, io
from phonecal.general import gaussMd

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
results_readnoise = results/"readnoise"

isos, stds  = io.load_stds  (folder, retrieve_value=io.split_iso)
colours     = io.load_colour(stacks)

s = stds[isos.argmin()]

RGBG, _ = raw.pull_apart(s, colours)

unique_colours = np.unique(colours).shape[0]

RGBG2 = np.stack([s for i in range(unique_colours)])
for j in np.unique(colours):
    RGBG2[j][colours != j] = np.nan

V = RGBG2.copy()
V[np.isnan(RGBG2)] = 0
VV = gaussMd(V, sigma=(0,10,10))

W = np.ones_like(RGBG2)
W[np.isnan(RGBG2)] = 0
WW = gaussMd(W, sigma=(0,10,10))

RGBG2 = VV/WW

def gauss_nan(data, *args, **kwargs):
    V = data.copy()
    V[np.isnan(data)] = 0
    VV = gaussMd(V, *args, **kwargs)

    W = np.ones_like(data)
    W[np.isnan(data)] = 0
    WW = gaussMd(W, *args, **kwargs)

    return VV/WW
