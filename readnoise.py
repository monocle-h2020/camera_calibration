import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import raw, plot, io
from phonecal.general import gaussMd

folder = argv[1]
isos, stds  = io.load_stds  (folder, retrieve_value=io.split_iso, file=True)
colours     = io.load_colour(folder)

low_iso = isos.argmin()
high_iso= isos.argmax()
