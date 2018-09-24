import numpy as np
from sys import argv
from ispex import io
from glob import glob
from os import walk

folder = argv[1]
subfolders = next(walk(folder))[1]
folders = [folder] + [f"{folder}/{sub}" for sub in subfolders]

for fol in folders:
    print(fol)
    DNGs = glob(f"{fol}/*.dng")
    if len(DNGs) == 0:
        continue
    arrs, colors = io.load_dng_many(fol+"/*.dng", return_colors=True)
    mean = arrs.mean(axis=0, dtype=np.float32)
    stds = arrs.std (axis=0, dtype=np.float32)

    name = fol.strip("/")
    np.save(f"{name}_mean.npy", mean)
    np.save(f"{name}_stds.npy", stds)
    np.save(f"{name}_colr.npy", colors)