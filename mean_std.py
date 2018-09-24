import numpy as np
from sys import argv
from ispex import io
from glob import glob
from os import walk

folder = argv[1]

for tup in walk(folder):
    fol = tup[0]
    name = fol.strip("/")

    DNGs = glob(f"{fol}/*.dng")
    if len(DNGs) == 0:
        continue
    
    print(f"{fol}  -->  {name}_x.npy")

    arrs, colors = io.load_dng_many(fol+"/*.dng", return_colors=True)
    mean = arrs.mean(axis=0, dtype=np.float32)
    stds = arrs.std (axis=0, dtype=np.float32)

    np.save(f"{name}_mean.npy", mean)
    np.save(f"{name}_stds.npy", stds)
    np.save(f"{name}_colr.npy", colors)
