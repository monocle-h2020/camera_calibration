import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import raw, io, plot, flat
from spectacle.general import gaussMd

meanfile = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(meanfile)
phone = io.load_metadata(root)

colours = io.load_colour(stacks)

iso = 23
exposure_time = 1/3

try:
    mean = np.load(meanfile)
except OSError:
    mean = io.load_raw_image(meanfile)

bias = np.load(products/"bias.npy")
dark = np.load(products/"dark.npy")

corrected_ADU = mean - bias - dark * exposure_time  # ADU
print("Corrected for bias and dark current")

ISO_model = io.read_iso_model(products)
iso_normalization = ISO_model(iso)

corrected_exposure = (phone["camera"]["f-number"]**2 / (exposure_time * iso_normalization) ) * corrected_ADU  # norm. ADU sr^-1 s^-1
print("Corrected for exposure parameters")

corrected_edges = corrected_exposure[flat.clip_border]
colours_edges = colours[flat.clip_border]
flat_field_correction = io.read_flat_field_correction(products, corrected_edges.shape)

corrected_flat = flat_field_correction * corrected_edges  # norm. ADU sr^-1 s^-1
print("Corrected for flat-field")

pixel_area_m = (phone["camera"]["pixel_size"] * 1e-6)**2
corrected_pixel_size = corrected_flat / pixel_area_m  # norm. ADU m^-2 sr^-1 s^-1
print("Corrected for pixel size")

effective_bandwidths = io.read_spectral_bandwidths(products) * 1e-9  # m
corrected_bandwidth = raw.multiply_RGBG(corrected_pixel_size, colours_edges, 1/effective_bandwidths)  # norm. ADU m^-2 sr^-1 s^-1 m^-1
print("Corrected for effective spectral bandwidths")

hc = 1.9864459e-25  # J m

corrected_RRU = hc * corrected_bandwidth  # RRU m^-2 sr^-1
print("Converted to energy")

corrected_RRU_RGBG,_ = raw.pull_apart(corrected_RRU, colours_edges)
plot.show_RGBG(corrected_RRU_RGBG)
