import numpy as np
from sys import argv
from spectacle import io
from os import walk, makedirs

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
camera = io.load_metadata(root)

raw_pattern = f"*{camera.image.raw_extension}"

for tup in walk(folder):
    folder_here = io.Path(tup[0])
    goal = io.replace_word_in_path(folder_here, "images", "stacks")

    raw_files = list(folder_here.glob(raw_pattern))
    if len(raw_files) == 0:
        continue

    arrs, colors = io.load_raw_image_multi(folder_here, pattern=raw_pattern)
    mean = arrs.mean(axis=0, dtype=np.float32)
    stds = arrs.std (axis=0, dtype=np.float32)

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
