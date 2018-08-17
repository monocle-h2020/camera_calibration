import numpy as np
from sys import argv
from shutil import move
from matplotlib import pyplot as plt
from ispex import raw, plot, io
from ispex.general import bin_centers, weighted_mean
from ispex.gamma import malus, malus_error
from glob import glob
import os
from time import time

folder_main = argv[1]
blocksize = int(argv[2])
files = glob(folder_main + "/*.dng")
files = sorted(files)
blocks = len(files) // blocksize
for i in range(blocks):
    foldername = str(int(time()*10000)) + str(i%10)
    total_path = f"{folder_main}/{foldername}"
    print(total_path)
    os.mkdir(total_path)
    files_block = files[blocksize*i : blocksize*(i+1)]
    for file in files_block:
        move(file, total_path)
        withjpg = os.path.splitext(file)[0] + ".jpg"
        try:
            move(withjpg, total_path)
        except FileNotFoundError:
            continue