"""
Analyse a flat-field data set.

Command line arguments:
    * `file`: the location of the mean flat-field data stack to be analysed.
"""

import numpy as np
from sys import argv
from spectacle import io, flat, calibrate, analyse
from spectacle.general import gaussMd
from matplotlib import pyplot as plt

# Get the data folder from the command line
meanfile = io.path_from_input(argv)
root = io.find_root_folder(meanfile)
savefolder = root/"analysis/flatfield/"
label = meanfile.stem.split("_mean")[0]

# Get metadata
camera = io.load_metadata(root)
print("Loaded metadata")

# Load the data
stdsfile = meanfile.parent / meanfile.name.replace("mean", "stds")
mean_raw = np.load(meanfile)
stds_raw = np.load(stdsfile)
print("Loaded data")

# Bias correction
mean_bias_corrected = calibrate.correct_bias(root, mean_raw)

# Normalise the RGBG2 channels to a maximum of 1 each
mean_normalised, stds_normalised = flat.normalise_RGBG2(mean_bias_corrected, stds_raw, camera.bayer_map)
print("Normalised data")

# Calculate the signal-to-noise ratio (SNR) per pixel
SNR = mean_normalised / stds_normalised
print("Calculated signal-to-noise-ratio")

# Make a histogram of the SNR
save_to_histogram_SNR = savefolder/f"data_histogram_SNR_{label}.pdf"
SNR_top_percentile = analyse.symmetric_percentiles(SNR)[1]
bins_SNR = np.linspace(0, SNR_top_percentile, 100)

plt.figure(figsize=(4,2), tight_layout=True)
plt.hist(SNR.ravel(), bins=bins_SNR, color='k')
plt.xlim(bins_SNR[0], bins_SNR[-1])
plt.xlabel("Signal-to-noise ratio")
plt.ylabel("Counts")
plt.savefig(save_to_histogram_SNR)
plt.close()
print(f"Saved histogram of signal-to-noise ratio to '{save_to_histogram_SNR}'")

# Make Gaussian maps of the SNR
save_to_maps_SNR = savefolder/f"data_SNR_{label}.pdf"
analyse.plot_gauss_maps(SNR, camera.bayer_map, colorbar_label="Signal-to-noise ratio", saveto=save_to_maps_SNR)
print(f"Saved maps of signal-to-noise ratio to '{save_to_maps_SNR}'")

# Convolve the flat-field data with a Gaussian kernel to remove small-scale variations
flat_field_gauss = gaussMd(mean_normalised, 10)

# Make histograms of the raw, bias-corrected, normalised, and Gaussed data
save_to_histogram_data = savefolder/f"data_histogram_{label}.pdf"
data_sets = [mean_raw, mean_normalised, mean_bias_corrected, flat_field_gauss]
titles = ["Raw", "Normalised", "Bias-corrected", "Gaussed"]
bins_adu = np.linspace(0, camera.saturation, 250)
bins_flat = np.linspace(0, 1.05, 100)
bins_combined = [bins_adu, bins_flat, bins_adu, bins_flat]
fig, axs = plt.subplots(nrows=2, ncols=2, tight_layout=True, sharex="col", sharey="col")
for data, ax, title, bins in zip(data_sets, axs.ravel(), titles, bins_combined):
    ax.hist(data.ravel(), bins=bins, color='k')
    ax.grid(True, ls="--")
    ax.set_title(title)
    ax.set_xlim(bins[0], bins[-1])
    ax.set_ylabel("Counts")
for ax in axs[:,1]:
    ax.tick_params(axis='y', left=False, labelleft=False, right=True, labelright=True)
    ax.yaxis.set_label_position("right")
axs[1,0].set_xlabel("Digital value (ADU)")
axs[1,1].set_xlabel("Correction factor")
plt.savefig(save_to_histogram_data)
plt.close()
print(f"Saved histograms of data to '{save_to_histogram_data}'")

# Make a histogram of the difference between the normalised and Gaussed data
save_to_histogram_difference = savefolder/f"data_gauss_difference.pdf"
difference_normalised_gauss = flat_field_gauss - mean_normalised
bins = np.linspace(*analyse.symmetric_percentiles(difference_normalised_gauss, percent=0.01), 100)
plt.figure(figsize=(4,2), tight_layout=True)
plt.hist(difference_normalised_gauss.ravel(), bins=bins, color='k')
plt.xlim(bins[0], bins[-1])
plt.xlabel("Gaussed map $-$ Normalised map")
plt.ylabel("Counts")
plt.savefig(save_to_histogram_difference)
plt.close()
print(f"Saved histogram of difference (Gaussed - Normalised data) to '{save_to_histogram_difference}'")

# Plot Gaussian maps of the flat-field data
save_to_maps_normalised = savefolder/f"data_normalised_{label}.pdf"
save_to_maps_gaussed = savefolder/f"data_gaussed_{label}.pdf"
analyse.plot_gauss_maps(mean_normalised, camera.bayer_map, colorbar_label="Flat-field response", vmax=1, saveto=save_to_maps_normalised)
analyse.plot_gauss_maps(flat_field_gauss, camera.bayer_map, colorbar_label="Flat-field response", vmax=1, saveto=save_to_maps_gaussed)
print(f"Saved Gaussed maps to '{save_to_maps_normalised}' and '{save_to_maps_gaussed}'")

# Clip the data
flat_field_gauss_clipped = flat.clip_data(flat_field_gauss)
colours_clipped = flat.clip_data(camera.bayer_map)

# Convert the data to a correction factor g
correction_factor = 1 / flat_field_gauss_clipped

# Plot Gaussian maps of the correction factors
save_to_maps_correction_factor = savefolder/f"data_correction_factor_{label}.pdf"
analyse.plot_gauss_maps(correction_factor, colours_clipped, colorbar_label="Correction factor", saveto=save_to_maps_correction_factor)
print(f"Saved correction factor maps to '{save_to_maps_correction_factor}'")
