import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from ispex import raw, plot, io
from ispex.general import bin_centers
from glob import glob
from scipy.stats import binned_statistic

mainfolder = argv[1]
subfolders = glob(f"{mainfolder}/*")
iso = int(mainfolder.split("iso")[1].strip("/"))

meanRGB = [[], [], []]
varRGB  = [[], [], []]

for folder in subfolders:
    print(folder)
    arrs, colors = io.load_dng_many(f"{folder}/*.dng", return_colors=True)
    mean = arrs.mean(axis=0) # mean per x,y
    var  = arrs.var(axis=0)
    RGBG_var, _ = raw.pull_apart(var, colors)
    RGBG_mean, _ = raw.pull_apart(mean, colors)
    for j in range(4):
        ind = 1 if j == 3 else j
        meanRGB[ind].append(RGBG_mean[..., j])
        varRGB[ind].append(RGBG_var[..., j])

C_range = np.arange(0, 4096, 1)
for j in range(3):
    colour = "RGB"[j]
    print(colour)
    mean = np.array(meanRGB[j]).ravel()
    var  = np.array(varRGB[j]).ravel()

    mean_per_I, bin_edges, bin_number = binned_statistic(mean, var, statistic="mean", bins=C_range)
    std_per_I = binned_statistic(mean, var, statistic=np.std, bins=C_range).statistic
    nr_per_I = binned_statistic(mean, var, statistic="count", bins=C_range).statistic
    bc = bin_centers(bin_edges)

    res = np.array([bc, nr_per_I, mean_per_I, std_per_I])
    np.save(f"results/gain/curves/iso_{iso}_{colour}.npy", res)
