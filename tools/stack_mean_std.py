"""
Walk through a folder and create NPY stacks based on the images found. This
script will walk through all the subfolders of a given folder and generate
NPY stacks one level above the lowest level found. For example, in a given file
structure `level1/level2/level3/image1.dng`, stacks will be generated at
`level1/level2/level3_mean.npy` and `level1/level2/level3_stds.npy`.

Images are assumed to be in the `images` subfolder (see `data_template`).

By default, the save folder is the same as the data folder, but with `images`
replaced with `stacks`.

For particularly large data sets, use the `stack_heavy.py` script instead.

Command line arguments:
    * `folder`: folder containing data. Any RAW (and optionally JPEG) images in
    this folder and any of its subfolders will be stacked, as described above.

TO DO:
    * Allow input/output folders that are not in `images` or `stacks`
"""
from os import walk, makedirs
import numpy as np
from spectacle import io, tqdm

# Command-line arguments
import argparse
parser = argparse.ArgumentParser(description="Walk through a folder and create NPY stacks based on the images found.")
parser.add_argument("folder", help="Folder containing data and/or subfolders with data.", type=io.Path)
parser.add_argument("-f", "--file_extension", help="RAW file extension, e.g. `dng` or `CR2`. Default: determine automatically.", default=None)
parser.add_argument("--skip_jpeg", help="Skip JPEG files (if present).", action="store_true")
parser.add_argument("-v", "--verbose", help="Enable verbose output (disables progress bar).", action="store_true")
args = parser.parse_args()
# TO DO: Option for heavy data sets - merge with stack_heavy.py (-H flag or similar)

# Common RAW file extensions - try all, then select the one that works (unless specified by the user)
raw_patterns = ["*.dng", "*.NEF", "*.CR2"]  # TO DO: Move this to the module level
if args.file_extension is None:
    raw_pattern = args.file_extension
    if args.verbose:
        print("No file format specified; to be determined automatically.")
else:
    raw_pattern = f"*.{args.file_extension}"
    if args.verbose:
        print(f"Will look for files that look like {raw_pattern}")

# Iterator; shows a general progress bar if not running in verbose mode
walker = tqdm(list(walk(args.folder)), desc="Processing folders", unit="folder", disable=args.verbose)

if args.verbose:
    print("Starting...")
# Walk through the folder and all its subfolders
for tup in walker:
    # The current folder
    folder_here = io.Path(tup[0])
    if args.verbose:
        print("\n")
        print(folder_here)

    # The folder to save stacks to
    goal = io.replace_word_in_path(folder_here, "images", "stacks")

    # If the correct RAW file format has not been determined yet, do so
    if raw_pattern is None:
        for pattern in raw_patterns:  # Looping over the patterns, look for any files matching it
            raw_files = list(folder_here.glob(pattern))
            if len(raw_files) > 0:  # If a match was found, set the raw pattern to match it, and break the loop
                raw_pattern = pattern
                if args.verbose:
                    print(f"Found a file format: {raw_pattern}")
                break
            # If no match was found, continue
        else:  # If no match was found at all, continue to the next folder
            if args.verbose:
                print("No files matching any known RAW formats found.")
            continue

    # Find all RAW files in this folder
    raw_files = list(folder_here.glob(raw_pattern))
    if len(raw_files) == 0:
        # If there are no RAW files in this folder, move on to the next
        if args.verbose:
            print(f"No RAW ({raw_pattern}) files found.")
        continue

    # Create the goal folder if it does not exist yet
    makedirs(goal.parent, exist_ok=True)
    savemean, savestds, savejmean, savejstds = [io.Path(f"{goal}_{stack}.npy") for stack in ("mean", "stds", "jmean", "jstds")]

    # Load all RAW files
    arrs = io.load_raw_image_multi(folder_here, pattern=raw_pattern, leave_progressbar=args.verbose)
    if args.verbose:
        print(f"Loaded RAW ({raw_pattern}) data")

    # Calculate and save the mean per pixel
    mean = arrs.mean(axis=0, dtype=np.float32)
    np.save(savemean, mean)
    del mean  # Clear up some memory
    if args.verbose:
        print(f"    -->  {savemean.absolute()}")

    # Calculate and save the standard deviation per pixel
    stds = arrs.std(axis=0, dtype=np.float32)
    np.save(savestds, stds)
    del stds, arrs  # Clear up some memory
    if args.verbose:
        print(f"    -->  {savestds.absolute()}")

    # Stop here if JPEGs are to be skipped
    if args.skip_jpeg:
        continue

    # Find all JPEG files in this folder
    JPGs = list(folder_here.glob("*.jp*g"))
    if len(JPGs) == 0:
        # If there are no JPEG files in this folder, move on to the next
        continue

    # Load all JPEG files
    jarrs = io.load_jpg_multi(folder_here, leave_progressbar=args.verbose)
    if args.verbose:
        print("Loaded JPEG data")

    # Calculate and save the mean per pixel
    jmean = jarrs.mean(axis=0, dtype=np.float32)
    np.save(savejmean, jmean)
    del jmean  # Clear up some memory
    if args.verbose:
        print(f"    -->  {savejmean.absolute()}")

    # Calculate and save the standard deviation per pixel
    jstds = jarrs.std(axis=0, dtype=np.float32)
    np.save(savejstds, jstds)
    del jstds, jarrs  # Clear up some memory
    if args.verbose:
        print(f"    -->  {savejstds.absolute()}")
