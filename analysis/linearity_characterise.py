"""
Analyse Pearson r (linearity) maps generated using the calibration scripts.
This script generates map images and histograms.

Command line arguments:
    * `file_raw`: the file containing the Pearson r map to be analysed. This r
    map should be an NPY stack generated using linearity_raw.py.
    Optional:
    * `file_jpeg`: the file containing the JPEG Pearson r map to be analysed.
    This r map should be an NPY stacks generated using linearity_jpeg.py.
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, analyse, plot

# Get the data folder from the command line
# Use JPEG data if these are provided
try:
    file_raw, file_jpeg = io.path_from_input(argv)
except TypeError:
    file_raw = io.path_from_input(argv)
    jpeg_data_available = False
else:
    jpeg_data_available = True
root = io.find_root_folder(file_raw)
if jpeg_data_available:
    print("JPEG data have been provided")
else:
    print("JPEG data are not available")

# Get metadata
camera = io.load_metadata(root)
savefolder = root/"analysis/linearity/"

# Load the data
r_raw = np.load(file_raw)
print("Loaded RAW Pearson r map")
if jpeg_data_available:
    r_jpeg = np.load(file_jpeg)
    print("Loaded JPEG Pearson r map")

# Make Gaussian maps of the RAW data
save_to_maps = savefolder/f"map_raw.pdf"
camera.plot_gauss_maps(r_raw, colorbar_label="Pearson $r$", saveto=save_to_maps)
print(f"Saved maps of RAW Pearson r to '{save_to_maps}'")

# Make a Gaussian map of the JPEG data, if available
if jpeg_data_available:
    save_to_map_JPEG = savefolder/f"map_jpeg.pdf"
    vmin, vmax = analyse.symmetric_percentiles(r_jpeg)
    fig, axs = plt.subplots(ncols=3, sharex=True, sharey=True, figsize=(5,2), tight_layout=True, gridspec_kw={"wspace":0, "hspace":0}, squeeze=True)
    for data, channel, ax in zip(r_jpeg, plot.rgb, axs):
        img = ax.imshow(data, vmin=vmin, vmax=vmax, cmap=plot.cmaps[channel])
        colorbar_here = plot.colorbar(img)
        ax.set_xticks([])
        ax.set_yticks([])
        if ax is axs[1]:
            colorbar_here.set_label("Pearson $r$")
        colorbar_here.locator = plot.ticker.MaxNLocator(nbins=4)
        colorbar_here.update_ticks()
    plt.savefig(save_to_map_JPEG)
    plt.close()
    print(f"Saved map of JPEG Pearson r to '{save_to_map_JPEG}'")

# Print comparative statistics on the RAW and JPEG (R/G/B) Pearson r values
if jpeg_data_available:
    r_all = np.stack([r_raw, *r_jpeg])
    labels = ["RAW", "JPEG R", "JPEG G", "JPEG B"]
    stats = analyse.statistics(r_all, prefix_column=labels, prefix_column_header="Channel")
    print(stats)

# Make an RGB histogram of the RAW r values
save_to_histogram_RGB = savefolder/f"histogram_RGB_raw.pdf"
xmax = 1.
camera.plot_histogram_RGB(r_raw, xmax=xmax, xlabel="Pearson $r$", saveto=save_to_histogram_RGB)
print(f"Saved RGB histogram to '{save_to_histogram_RGB}'")

# Make a histogram comparing RAW and JPEG r values
if jpeg_data_available:
    save_to_histogram_RAW_JPEG = savefolder/f"histogram_raw_jpeg.pdf"
    bins = np.linspace(0.9, 1.0, 150)
    plt.figure(tight_layout=True, figsize=(5,2))
    plt.hist(r_raw.ravel(), bins=bins, color='k')
    for j, c in enumerate("rgb"):
        plt.hist(r_jpeg[j].ravel(), bins=bins, color=c, alpha=0.7)
    plt.xlabel("Pearson $r$")
    plt.ylabel("Frequency")
    plt.yscale("log")
    plt.xlim(0.9, 1)
    plt.ylim(ymin=0.9)
    plt.xticks(rotation=30)
    plt.savefig(save_to_histogram_RAW_JPEG)
    plt.close()
    print(f"Saved RAW/JPEG histogram to '{save_to_histogram_RAW_JPEG}'")
