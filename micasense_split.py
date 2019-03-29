from sys import argv
from shutil import move
import os
from time import time
from phonecal import io

folder_main = io.path_from_input(argv[:2])
root, images, stacks, products, results = io.folders(folder_main)
phone = io.read_json(root/"info.json")
raw_pattern = f"*{phone['software']['raw extension']}"

files = list(folder_main.glob(raw_pattern))
files = sorted(files)

cameras = list(range(1, 6))
for cam in cameras:
    files_cam = [file for file in files if f"_{cam}.tif" in file.name]
    total_path = folder_main / str(cam)
    os.mkdir(total_path)
    for file in files_cam:
        print(file, "-->", total_path / file.name)
        move(file, total_path / file.name)