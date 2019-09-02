# Tools

This folder contains auxiliary scripts, such as for moving or renaming image files.

## File splitting

[split_files.py](split_files.py) may be used to split large amounts of RAW and/or JPEG files into subfolders of a given size. These can then be manually reorganised and/or renamed according to the conditions they correspond to. For example, one might take 10 images each at 10 different ISO speeds, use this script to split them into 10 subfolders, and then rename each subfolder according to its respective ISO speed.

## Image stacking

Many of the SPECTACLE calibration and analysis scripts are based on image statistics, such as the mean or standard deviation value per pixel when taking multiple identical exposures. [stack_mean_std.py](stack_mean_std.py) is used to generate such image stacks (in NPY format) from a folder structure containing RAW files
