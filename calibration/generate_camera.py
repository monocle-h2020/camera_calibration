"""
Generate a camera information file for a camera given a RAW image and user inputs on the command line.
Ideally, the given image is one with saturated pixels.

Note that if the image does not have at least one saturated pixel, the user will be asked to input the bit depth manually.

Command line arguments:
    * filename: location of a RAW image from which the data can be extracted.

Example:
    python calibration/generate_camera.py ~/SPECTACLE_data/iPhone_SE/images/IMG_0001.DNG

To do:
    * Generate entire file structure for camera - or separate script?
"""
from spectacle import io
from spectacle.camera import Camera, name_from_root_folder

# Command-line arguments
import argparse
parser = argparse.ArgumentParser(description="Generate a camera metadata file from a RAW image.")
parser.add_argument("filename", help="Location of RAW image.", type=io.Path)
parser.add_argument("-o", "--output_folder", help="Folder to save metadata file to (default: input root folder).", type=io.Path, default=None)
parser.add_argument("-v", "--verbose", help="Enable verbose output.", action="store_true")
args = parser.parse_args()

# Get the data folder from the command line
root = io.find_root_folder(args.filename)

# Get the data
raw_file = io.load_raw_file(args.filename)
if args.verbose:
    print("Loaded raw file")
exif = io.load_exif(args.filename)
if args.verbose:
    print("Loaded exif metadata")

# Get the camera name from the root folder, then ask the user for feedback
camera_name = name_from_root_folder(root)
camera_name_correct = input(f"From the image location (`{args.filename.absolute()}`),\nthe camera name was determined to be '{camera_name}'. Is this correct? [y/n]\n")
if camera_name_correct not in ("Y", "y", "yes", "Yes"):
    camera_name = input("Please put in the correct name for this camera:\n")

# Bit depth - find the maximum value and the corresponding bit depth
maximum_value = raw_file.raw_image.max()
bit_depth_conversion = {255: 8, 511: 9, 1023: 10, 2047: 11, 4095: 12, 8191: 13,
                        16383: 14, 32767: 15, 65535: 16}
try:
    bit_depth = bit_depth_conversion[maximum_value]
except KeyError:
    print(f"The provided image ({args.filename}) does not have any saturated pixels (maximum value: {maximum_value}).")
    bit_depth = input("Please enter the bit depth manually.\nTypical values are 8 (JPEG), 10 (Samsung), 12 (Apple), 14 (Nikon).\n")
    bit_depth = int(bit_depth)

# Camera properties
properties = {
        "name": camera_name,
        "manufacturer": exif["Image Make"].printable,
        "name_internal": exif["Image Model"].printable,
        "image_shape": raw_file.raw_image.shape,
        "raw_extension": args.filename.suffix,
        "bias": raw_file.black_level_per_channel,
        "bayer_pattern": raw_file.raw_pattern.tolist(),
        "bit_depth": bit_depth,
        "colour_description": raw_file.color_desc.decode()
        }
if args.verbose:
    print("Camera properties:", properties)

# Combine all metadata into a single object
camera = Camera(**properties, root=root)

# Save to a filename based on the camera name
# Can be overridden by the --output_folder argument
savefolder = camera.root if args.output_folder is None else args.output_folder
save_to = savefolder/f"{camera.name_underscore}_data.json"

# Save the Camera object to file
camera.write_to_file(save_to)
print(f"Saved metadata to '{save_to}'")
