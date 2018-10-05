import numpy as np
from sys import argv
from phonecal import io
from glob import glob
from os import walk, path

folder_images = argv[1]
folder_stacks = folder_images.replace("images", "stacks")
colour_path = folder_stacks.split("stacks")[0] + "/stacks/colour.npy"
colour_exists = path.exists(colour_path)

for tup in walk(folder_images):
    folder_here = tup[0]
    goal = folder_here.strip("/").replace("images", "stacks")

    DNGs = glob(f"{folder_here}/*.dng")
    if len(DNGs) == 0:
        continue
    
    print(f"{folder_here}  -->  {goal}_x.npy")

    arrs, colors = io.load_dng_many(f"{folder_here}/*.dng", return_colors=True)
    mean = arrs.mean(axis=0, dtype=np.float32)
    stds = arrs.std (axis=0, dtype=np.float32)

    np.save(f"{goal}_mean.npy", mean)
    np.save(f"{goal}_stds.npy", stds)
    if not colour_exists:
        np.save(colour_path, colors)
        colours_exists = True

    JPGs = glob(f"{folder_here}/*.jp*g")
    if len(JPGs) == 0:
        continue
    
    jarrs = io.load_jpg_many(f"{folder_here}/*.jp*g")
    jmean = jarrs.mean(axis=0, dtype=np.float32)
    jstds = jarrs.std (axis=0, dtype=np.float32)
    
    np.save(f"{goal}_jmean.npy", jmean)
    np.save(f"{goal}_jstds.npy", jstds)
