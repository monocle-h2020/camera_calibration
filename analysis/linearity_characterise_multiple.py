"""
Analyse Pearson r (linearity) maps for multiple cameras, generated using the
calibration scripts. This script returns some statistics on each and generates
a histogram comparing them.

This script assumes the r maps to be located at
root/"intermediaries/linearity/linearity_raw.npy"
If root/"intermediaries/linearity/linearity_jpeg.npy" exists, JPEG r values are
also included in the histogram.

Command line arguments:
    * `folders`: folders containing the Pearson r maps. These r maps should be
    NPY stacks generated using linearity_raw.py and/or linearity_jpeg.py.
    (multiple arguments possible)
"""

import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, linearity as lin, plot

# Get the data folder from the command line
folders = io.path_from_input(argv)
roots = [io.find_root_folder(folder) for folder in folders]
r_raw_paths = [root/"intermediaries/linearity/linearity_raw.npy" for root in roots]
r_jpeg_paths = [root/"intermediaries/linearity/linearity_jpeg.npy" for root in roots]
save_to = io.results_folder

# Load Camera objects
cameras = [io.load_camera(root) for root in roots]
print(f"Loaded Camera objects: {cameras}")

def load_jpeg(path):
    """
    Load JPEG data if available, else return None
    """
    try:
        jpeg = np.load(path)
    except FileNotFoundError:
        jpeg = None
    return jpeg

# Load the data
r_raw  = [  np.load(raw_path ) for raw_path  in r_raw_paths ]
r_jpeg = [load_jpeg(jpeg_path) for jpeg_path in r_jpeg_paths]

# Lower limit of the horizontal axis in the plot
lower_limit = 0.85

bins = np.linspace(lower_limit, 1.0, 150)

fig, axs = plt.subplots(nrows=len(r_raw), sharex=True, sharey=True, squeeze=True, tight_layout=True, figsize=(5.33,0.9*len(r_raw)), gridspec_kw={"wspace":0, "hspace":0})
for ax, raw_, jpeg_, camera in zip(axs, r_raw, r_jpeg, cameras):
    # Remove NaN elements
    raw = raw_[~np.isnan(raw_)]
    print(camera)
    below_limit = np.where(raw < lower_limit)[0]
    print(f"RAW    pixels with r < {lower_limit}: {len(below_limit):>3}")
    ax.hist(raw.ravel(), bins=bins, color='k', edgecolor="None")
    if jpeg_ is not None:
        for j_, c in zip(jpeg_, plot.RGB_OkabeIto):
            jpeg_c = j_[~np.isnan(j_)]
            below_limit = np.where(jpeg_c < lower_limit)[0]
            print(f"JPEG {c} pixels with r < {lower_limit}: {len(below_limit):>3}")
            ax.hist(jpeg_c.ravel(), bins=bins, color=c, edgecolor="None", linewidth=0, alpha=0.7)
    ax.text(0.025, 0.85, camera.name, bbox=plot.bbox_text, transform=ax.transAxes, horizontalalignment="left", verticalalignment="top")
    ax.axvline(lin.linearity_limit, c='k', ls="--")
    bad_pixels = np.where(raw < lin.linearity_limit)[0]
    print(f"RAW    pixels with r < {lin.linearity_limit}: {len(bad_pixels):>3}\n")
axs[0].set_xlim(lower_limit, 1)
axs[0].set_yscale("log")
axs[0].set_ylim(ymin=10)
for ax in axs:
    ax.set_yticks([1e2, 1e4, 1e6])
axs[len(r_raw)//2].set_ylabel("Number of pixels")
axs[-1].set_xlabel("Pearson $r$")
fig.savefig(save_to/"linearity_comparison.pdf")
plt.show()
plt.close()
