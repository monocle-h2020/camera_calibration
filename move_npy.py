import shutil, os
from glob import glob
from sys import argv
import time

origin = argv[1]
dest   = argv[2]

name = origin.strip("/").split("/")[-1]

for tup in os.walk(origin):
    fol = tup[0]
    name = fol.strip("/")
    NPYs = glob(f"{name}/*.npy")
    if not NPYs:
        continue
    relname = os.path.relpath(name, origin)
    newdest = f"{dest}/{relname}"
    os.makedirs(newdest)
    for npy_file in NPYs:
        shutil.copy(npy_file, newdest)
    print(name)
