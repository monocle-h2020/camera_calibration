import numpy as np
from sys import argv
from spectacle import io
from os import walk, makedirs

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")

colour_path = stacks/"colour.npy"
colour_exists = colour_path.exists()

raw_pattern = f"*{phone['software']['raw extension']}"

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
    if not colour_exists:
        np.save(colour_path, colors)
        colours_exists = True

    print(f"{folder_here}  -->  {goal}_x.npy")

    JPGs = list(folder_here.glob("*.jp*g"))
    if len(JPGs) == 0:
        continue

    jarrs = io.load_jpg_many(folder_here)
    jmean = jarrs.mean(axis=0, dtype=np.float32)
    jstds = jarrs.std (axis=0, dtype=np.float32)

    np.save(f"{goal}_jmean.npy", jmean)
    np.save(f"{goal}_jstds.npy", jstds)
