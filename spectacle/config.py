"""
Configuration file for SPECTACLE.

Edit these variables if you need.

This file is included in .gitignore so you do not accidentally push it to GitHub.
"""

from pathlib import Path

# Do not edit these
home_folder = Path.home()



# Edit these with your information if you wish
# folder containing all SPECTACLE data
spectacle_folder = home_folder / "SPECTACLE_data"

# folder containing results for inter-camera comparisons
# note that results for individual cameras are stored in `spectacle_folder/camera/results/`
results_folder = home_folder / "SPECTACLE_results"
