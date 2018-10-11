from sys import argv
from shutil import move
import os
from time import time
from phonecal import io

folder_main = io.path_from_input(argv[:2])
blocksize = int(argv[2])
files = list(folder_main.glob("*.dng"))
files = sorted(files)
blocks = len(files) // blocksize
for i in range(blocks):
    foldername = str(int(time()*10000)) + str(i%10)
    total_path = folder_main/foldername
    print(total_path)
    os.mkdir(total_path)
    files_block = files[blocksize*i : blocksize*(i+1)]
    for file in files_block:
        move(str(file), total_path)
        withjpg = io.replace_suffix(file, ".jpg")
        try:
            move(str(withjpg), total_path)
        except FileNotFoundError:
            continue
