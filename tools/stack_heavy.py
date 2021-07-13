"""
Walk through a folder and create NPY stacks based on the images found. This
script will walk through all the subfolders of a given folder and generate
NPY stacks one level above the lowest level found. For example, in a given file
structure `level1/level2/level3/image1.dng`, stacks will be generated at
`level1/level2/level3_mean.npy` and `level1/level2/level3_stds.npy`.

By default, image stacks are saved in the `root/stacks/` folder.

This script is intended for particularly large data sets. For smaller ones,
use the `stack_mean_std.py` script instead.

Command line arguments:
    * `folder`: folder containing data. Any RAW (and optionally JPEG) images in
    this folder and any of its subfolders will be stacked, as described above.

This script will be merged into `stack_mean_std.py`, so please refer to that
script for further documentation.
"""

import numpy as np
from sys import argv
from spectacle import io
from os import walk, makedirs

folder = io.path_from_input(argv)
root = io.find_root_folder(folder)

# Load Camera object
camera = io.load_camera(root)
print(f"Loaded Camera object: {camera}")

raw_pattern = f"*{camera.raw_extension}"

for tup in walk(folder):
    folder_here = io.Path(tup[0])
    goal = io.replace_word_in_path(folder_here, "images", "stacks")

    raw_files = list(folder_here.glob(raw_pattern))
    if len(raw_files) == 0:
        continue

    arrs = io.load_raw_image_multi(folder_here, pattern=raw_pattern)
    mean = np.full(arrs.shape[1:], np.nan)
    stds = mean.copy()
    for i, row in enumerate(mean):
        mean[i] = arrs[:,i].mean(axis=0, dtype=np.float32)
        stds[i] = arrs[:,i].std (axis=0, dtype=np.float32)
        if not i%100:
            print(f"{i/len(mean)*100:.1f}%")
    del arrs

    makedirs(goal.parent, exist_ok=True)

    np.save(f"{goal}_mean.npy", mean)
    np.save(f"{goal}_stds.npy", stds)

    print(f"{folder_here}  -->  {goal}_x.npy")

    JPGs = list(folder_here.glob("*.jp*g"))
    if len(JPGs) == 0:
        continue

    jarrs = io.load_jpg_multi(folder_here)
    jmean = jarrs.mean(axis=0, dtype=np.float32)
    jstds = jarrs.std (axis=0, dtype=np.float32)

    np.save(f"{goal}_jmean.npy", jmean)
    np.save(f"{goal}_jstds.npy", jstds)
