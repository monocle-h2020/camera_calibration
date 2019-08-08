"""
Configuration file for SPECTACLE.

Edit these variables if you need.

This file is included in .gitignore so you do not accidentally push it to GitHub.
"""

from pathlib import Path
from os import makedirs

# Do not edit these
home_folder = Path.home()

def create_folders():
    """
    Create the specified folders for data (spectacle_folder) and results (results_folder)
    """
    try:
        makedirs(spectacle_folder)
    except FileExistsError:
        print(f"Requested SPECTACLE data folder '{spectacle_folder}' already exists.")
    else:
        print(f"Created SPECTACLE data folder: {spectacle_folder}")

    try:
        makedirs(results_folder)
    except FileExistsError:
        print(f"Requested SPECTACLE results folder '{results_folder}' already exists.")
    else:
        print(f"Created SPECTACLE results folder: {results_folder}")


# Edit these with your information if you wish
# folder containing all SPECTACLE data
spectacle_folder = home_folder / "SPECTACLE_data"

# folder containing results for inter-camera comparisons
# note that results for individual cameras are stored in `spectacle_folder/camera/results/`
results_folder = home_folder / "SPECTACLE_results"
