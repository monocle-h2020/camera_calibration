"""
Walk through a folder and create NPY stacks based on the images found. This
script will walk through all the subfolders of a given folder and generate
NPY stacks one level above the lowest level found. For example, in a given file
structure `level1/level2/level3/image1.dng`, stacks will be generated at
`level1/level2/level3_mean.npy` and `level1/level2/level3_stds.npy`.

Images are assumed to be in the `images` subfolder (see `data_template`).

By default, image stacks are saved in the `root/stacks/` folder.

For particularly large data sets, use the `stack_heavy.py` script instead.

Command line arguments:
    * `folder`: folder containing data. Any RAW (and optionally JPEG) images in
    this folder and any of its subfolders will be stacked, as described above.

TO DO:
    * Allow input/output folders that are not in `images` or `stacks`
"""

import numpy as np
from sys import argv
from spectacle import io
from os import walk, makedirs

# Get the data folder from the command line
folder = io.path_from_input(argv)
root = io.find_root_folder(folder)
print("Loaded metadata")

# Get the camera metadata
camera = io.load_metadata(root)

# Wildcard pattern to find RAW data with
raw_pattern = f"*{camera.image.raw_extension}"

# Walk through the folder and all its subfolders
for tup in walk(folder):
    # The current folder
    folder_here = io.Path(tup[0])

    # The folder to save stacks to
    goal = io.replace_word_in_path(folder_here, "images", "stacks")

    # Find all RAW files in this folder
    raw_files = list(folder_here.glob(raw_pattern))
    if len(raw_files) == 0:
        # If there are no RAW files in this folder, move on to the next
        continue

    # Load all RAW files
    arrs = io.load_raw_image_multi(folder_here, pattern=raw_pattern)

    # Calculate the mean and standard deviation per pixel
    mean = arrs.mean(axis=0, dtype=np.float32)
    stds = arrs.std (axis=0, dtype=np.float32)

    # Create the goal folder if it does not exist yet
    makedirs(goal.parent, exist_ok=True)

    # Save the RAW data stacks
    np.save(f"{goal}_mean.npy", mean)
    np.save(f"{goal}_stds.npy", stds)

    # Print the input and output folder as confirmation
    print(f"{folder_here}  -->  {goal}_x.npy")

    # Find all JPEG files in this folder
    JPGs = list(folder_here.glob("*.jp*g"))
    if len(JPGs) == 0:
        # If there are no JPEG files in this folder, move on to the next
        continue

    # Load all JPEG files
    jarrs = io.load_jpg_multi(folder_here)

    # Calculate the mean and standard deviation per pixel
    jmean = jarrs.mean(axis=0, dtype=np.float32)
    jstds = jarrs.std (axis=0, dtype=np.float32)

    # Save the JPEG data stacks
    np.save(f"{goal}_jmean.npy", jmean)
    np.save(f"{goal}_jstds.npy", jstds)
