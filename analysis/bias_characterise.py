"""
Analyse bias maps (in ADU) generated using the calibration functions.

Command line arguments:
    * `folder`: the folder containing the bias maps to be analysed.
"""

from sys import argv
from spectacle import raw, plot, io, analyse
from spectacle.general import gaussMd

# Get the data folder from the command line
folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)

# Get metadata
phone = io.load_metadata(root)
colours = io.load_colour(stacks)
savefolder = results/"bias"

# Load the data
isos, means = io.load_means(folder, retrieve_value=io.split_iso)
print("Loaded data")

# Print statistics of the bias at each ISO
stats = analyse.statistics(means, prefix_column=isos, prefix_column_header="ISO")
print(stats)

# Range on the x axis for the histograms
xmax = phone["software"]["bias"] + 25
xmin = max(phone["software"]["bias"] - 25, 0)

# Loop over the data and make plots at each ISO value
for iso, mean in zip(isos, means):
    # Split the data into the Bayer RGBG2 filters and make an RGB histogram
    mean_RGBG, _ = raw.pull_apart(mean, colours)
    saveto_histogram = savefolder/f"histogram_iso{iso}.pdf"
    plot.histogram_RGB(mean_RGBG, xlim=(xmin, xmax), xlabel="Bias (ADU)", saveto=saveto_histogram, nrbins=100)
    print(f"Saved RGB histogram to '{saveto_histogram}'")
   
    # Convolve the maps with a Gaussian kernel
    gauss_combined = gaussMd(mean, sigma=10)
    gauss_RGBG = gaussMd(mean_RGBG, sigma=(0,5,5))
    
    # Find limits for the colorbar, to be used in every map image
    vmin, vmax = gauss_RGBG.min(), gauss_RGBG.max()
    
    # Plot images of the Gauss-convolved map
    saveto_map = savefolder/f"bias_map_iso{iso}.pdf"
    plot.show_image(gauss_combined, colorbar_label="Bias (ADU)", saveto=saveto_map)
    plot.show_image_RGBG2(gauss_RGBG, colorbar_label="Bias (ADU)", saveto=saveto_map, vmin=vmin, vmax=vmax)
    plot.show_RGBG(gauss_RGBG, colorbar_label=25*" "+"Bias (ADU)", saveto=savefolder/f"bias_map_iso{iso}_RGBG2.pdf", vmin=vmin, vmax=vmax)
    print(f"Saved RGBG2 maps to '{saveto_map}'")
