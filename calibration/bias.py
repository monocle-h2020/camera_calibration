"""
Create a bias map using the mean bias (zero-light, shortest-exposure images) images.
Bias data for all ISOs are loaded, but the map is only saved for the lowest ISO.

Command line arguments:
    * `folder`: folder containing NPY stacks of bias data taken at different ISO speeds.

Example:
    python calibration/bias.py ~/SPECTACLE_data/iPhone_SE/stacks/bias_readnoise/

To do:
    * Save maps for all ISOs and use these in the calibration process.
"""
import numpy as np
from spectacle import io

# Command-line arguments
import argparse
parser = argparse.ArgumentParser(description="Create a bias map using the stacked bias (zero-light, shortest-exposure images) images.")
parser.add_argument("folder", help="Folder containing .npy stacks of bias data.", type=io.Path)
parser.add_argument("-o", "--output_folder", help="Folder to save results files to (default: camera outputs folder).", type=io.Path, default=None)
parser.add_argument("-v", "--verbose", help="Enable verbose output.", action="store_true")
args = parser.parse_args()

# Get the root data folder
root = io.find_root_folder(args.folder)

# Load Camera object
camera = io.load_camera(root)
if args.verbose:
    print(f"Loaded Camera object: {camera}")

# Save location based on camera name (unless otherwise specified)
filename = "bias.npy"
save_to = camera.filename_calibration(filename) if args.output_folder is None else args.output_folder / filename

# Load the mean stacks for each ISO value
isos, means = io.load_means(args.folder, retrieve_value=io.split_iso, leave_progressbar=args.verbose)
if args.verbose:
    print(f"Loaded bias data for {len(isos)} ISO values from '{args.folder.absolute()}'")

# Find the lowest ISO value in the data and select the respective bias map
lowest_iso_index = isos.argmin()
bias_map = means[lowest_iso_index]

# Save the bias map for calibration purposes
np.save(save_to, bias_map)
print(f"\nSaved bias map at ISO {isos[lowest_iso_index]} to '{save_to}'")
