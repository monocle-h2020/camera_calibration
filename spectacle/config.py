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
    Create the specified folder for results (results_folder)
    """
    try:
        makedirs(results_folder)
    except FileExistsError:
        print(f"Requested SPECTACLE results folder '{results_folder}' already exists.")
    else:
        print(f"Created SPECTACLE results folder: {results_folder}")


# folder containing results for inter-camera comparisons
# note that results for individual cameras are stored in `spectacle_folder/camera/results/`
results_folder = home_folder / "SPECTACLE_results"
