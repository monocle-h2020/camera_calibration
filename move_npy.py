import shutil, os
from sys import argv
from phonecal import io

origin, dest = io.path_from_input(argv)

for tup in os.walk(origin):
    fol = tup[0]
    name = io.Path(fol)
    NPYs = list(name.glob("*.npy"))
    if not NPYs:
        continue
    relname = os.path.relpath(name, origin)
    newdest = dest/relname
    os.makedirs(newdest)
    for npy_file in NPYs:
        shutil.copy(npy_file, newdest)
    print(name)
