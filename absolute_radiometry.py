import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import raw, io, plot, flat
from phonecal.general import gaussMd

meanfile = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(meanfile)
phone = io.read_json(root/"info.json")

colours = io.load_colour(stacks)

iso = 23
exposure_time = 1/3

mean = np.load(meanfile)

bias = np.load(products/"bias.npy")
dark = np.load(products/"dark.npy")

corrected_ADU = mean - bias - dark * exposure_time  # ADU

ISO_model = io.read_iso_model(products)
iso_normalization = ISO_model(iso)

corrected_exposure = (phone["camera"]["f-number"]**2 / (exposure_time * iso_normalization) ) * corrected_ADU  # norm. ADU sr^-1 s^-1

corrected_edges = corrected_exposure[flat.clip_border]
colours_edges = colours[flat.clip_border]
flat_field_correction = io.read_flat_field_correction(products, corrected_edges.shape)

corrected_flat = flat_field_correction * corrected_edges  # norm. ADU sr^-1 s^-1

pixel_area_m = (phone["camera"]["pixel_size"] * 1e-6)**2
corrected_pixel_size = corrected_flat / pixel_area_m  # norm. ADU m^-2 sr^-1 s^-1

effective_bandwidths = io.read_spectral_bandwidths(products) * 1e-9  # m
corrected_bandwidth = raw.multiply_RGBG(corrected_pixel_size, colours_edges, 1/effective_bandwidths)  # norm. ADU m^-2 sr^-1 s^-1 m^-1

hc = 1.9864459e-25  # J m

corrected_RRU = hc * corrected_bandwidth
