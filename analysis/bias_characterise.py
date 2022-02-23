"""
Analyse bias maps (in ADU) generated using the calibration functions.

Command line arguments:
    * `folder`: folder containing NPY stacks of bias data taken at different ISO speeds.

Example:
    python analysis/bias_characterise.py ~/SPECTACLE_data/iPhone_SE/stacks/bias_readnoise/
"""
from sys import argv
from spectacle import io, analyse

# Get the data folder from the command line
folder = io.path_from_input(argv)
root = io.find_root_folder(folder)

# Load Camera objects
camera = io.load_camera(root)
print(f"Loaded Camera object: {camera}")

# Save location based on camera name
savefolder = camera.filename_analysis("bias", makefolders=True)

# Load the data
isos, means = io.load_means(folder, retrieve_value=io.split_iso)
print("Loaded data")

# Print statistics at each ISO
stats = analyse.statistics(means, prefix_column=isos, prefix_column_header="ISO")
print(stats)

# Range on the x axis for the histograms
xmin, xmax = analyse.symmetric_percentiles(means, percent=0.001)

# Loop over the data and make plots at each ISO value
for ISO, mean in zip(isos, means):
    save_to_histogram = savefolder/f"bias_histogram_iso{ISO}.pdf"
    save_to_maps = savefolder/f"bias_map_iso{ISO}.pdf"

    camera.plot_histogram_RGB(mean, xmin=xmin, xmax=xmax, xlabel="Bias (ADU)", saveto=save_to_histogram)
    camera.plot_gauss_maps(mean, colorbar_label="Bias (ADU)", saveto=save_to_maps)

    print(f"Saved plots for ISO speed {ISO}")
