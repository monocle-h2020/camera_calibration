import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import raw, io, plot
from phonecal.general import gaussMd

meanfile = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(meanfile)
phone = io.read_json(root/"info.json")

iso = 23
exposure_time = 1/3

mean = np.load(meanfile)

bias = np.load(products/"bias.npy")
dark = np.load(products/"dark.npy")

ADU_corrected = mean - bias - dark * exposure_time

ISO_model = io.read_iso_model(products)
iso_normalization = ISO_model(iso)
