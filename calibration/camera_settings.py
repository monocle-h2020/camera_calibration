"""
Generate a file with additional information for the camera in the `root` folder.

Command line arguments:
    * `folder`: folder containing camera data.

Example:
    python calibration/camera_settings.py ~/SPECTACLE_data/iPhone_SE/

To do:
    * Apertures?
"""
from spectacle import load_camera, io

# Command-line arguments
import argparse
parser = argparse.ArgumentParser(description="Generate a file with additional information for the camera.")
parser.add_argument("folder", help="Folder containing camera data.", type=io.Path)
parser.add_argument("-o", "--output_folder", help="Folder to save settings file to (default: Camera calibration folder).", type=io.Path, default=None)
parser.add_argument("-v", "--verbose", help="Enable verbose output.", action="store_true")
args = parser.parse_args()

# Get the data folder from the command line
root = io.find_root_folder(args.folder)

# Load Camera object
camera = load_camera(root)
if args.verbose:
    print(f"Loaded Camera object: {camera}")

# Save locations
filename_settings = "settings.json"
save_to = camera.filename_calibration(filename_settings) if args.output_folder is None else args.output_folder/filename_settings

# Get additional data from command line input from the user
iso_min = int(input("What is the *lowest* ISO speed available on this device? (-1 if unknown)\n"))
iso_max = int(input("What is the *highest* ISO speed available on this device? (-1 if unknown)\n"))

exposure_min = input("What is the *shortest* exposure time, in seconds, available on this device? (-1 if unknown)\nThis can be provided as an integer (e.g. 5 or 10), float (e.g. 0.12 or 5.1), or fraction (e.g. 1/5 or 2/3)\n")
exposure_max = input("What is the *longest* exposure time, in seconds, available on this device? (-1 if unknown)\nThis can be provided as an integer (e.g. 5 or 10), float (e.g. 0.12 or 5.1), or fraction (e.g. 1/5 or 2/3)\n")

print("")

# Settings
settings = {
        "ISO_min": iso_min,
        "ISO_max": iso_max,
        "exposure_min": exposure_min,
        "exposure_max": exposure_max}
if args.verbose:
    print("Camera settings:", settings)

# Combine the settings to file
io.write_json(settings, save_to)
print(f"Saved metadata to '{save_to}'")
