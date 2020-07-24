# SPECTACLE

SPECTACLE (Standardised Photographic Equipment Calibration Technique And CataLoguE) is a standardised methodology for the spectral and radiometric calibration of consumer camera data.
The associated database, containing calibration data for a number of popular consumer cameras, can be found at http://spectacle.ddq.nl/.
More information on the SPECTACLE methodology, including results from applying it to several cameras, can be found in our paper: https://doi.org/10.1364/OE.27.019075

This repository contains the associated `spectacle` Python module.
This module can be used to calibrate data using previously obtained calibration data (measured by the user or retrieved from the SPECTACLE database).
It also includes functions and pre-made scripts for processing calibration data, as described in the paper linked above.

# Installation

Currently, the easiest way to install the `spectacle` module is using `pip`: simply run `pip install pyspectacle` in your terminal to fetch the package from PyPI and install it.

You may have to specify a user-specific installation (`pip install pyspectacle --user`) if a permission error occurs.
Please note that while the module is identified as `pyspectacle` on PyPI and in pip, in Python itself it is imported and used as simply `spectacle`.

An alternative way to install the `spectacle` module is to clone this repository (`git clone git@github.com:monocle-h2020/camera_calibration.git`) and then install it using pip, by navigating into the repository folder and running `pip install .` (mind the `.`).

# Usage

There are three main use cases for the `spectacle` module, each of which will be explained further in the relevant subsection.
They are as follows:

1. Application: applying camera calibrations to new data.
2. Analysis: analysing camera properties and performance based on calibration data.
3. Calibration: generating calibration data for use in the two other use cases.

## Application

There are two interfaces for applying calibrations to new data, namely through a `spectacle.Camera` object or through the [`spectacle.calibrate`](spectacle/calibrate.py) submodule.

### spectacle.Camera interface

The `spectacle.Camera` interface is the easiest to use. 
A camera information file is generated using the [generate_camera.py](calibration/generate_camera.py) script.
This camera information file can be loaded in any script using the `spectacle.load_camera` function, which takes one argument, namely the `root` folder that contains all calibration data for a certain camera.

For example, if your calibration data for an iPhone SE are stored in the folder `/home/spectacle_data/iPhone_SE/`, then that folder is the `root` folder and the camera information file should be located in that folder (i.e. at `/home/spectacle_data/iPhone_SE/metadata.json`).
Then the Camera object can be initialised from that file and used in the future.

Calibrations are applied using the Camera object's methods, such as `Camera.correct_bias` for correcting for camera bias.
The Camera object will automatically load the required calibration data from the same folder it was initialised from.

Using the example of the iPhone SE, one might run the following piece of code:
```python3
from spectacle import load_metadata, io

camera = load_metadata("/home/spectacle_data/iPhone_SE/")
raw_data = io.load_raw_image("/home/img_0001.dng")

data_corrected = camera.correct_bias(raw_data)
```
This code snippet loads the iPhone SE camera data and a RAW image file (`/home/img_0001.dng`), then corrects the RAW image data for the iPhone SE camera bias.



### spectacle.calibrate interface

To use the `spectacle.calibrate` interface, simply load the [`spectacle.calibrate`](spectacle/calibrate.py) submodule (`from spectacle import calibrate`) and apply the methods contained therein.
For example, to correct for the camera bias, one would use the `correct_bias` method from this submodule.
Each method comes with detailed documentation on its usage, which can be found [here](spectacle/calibrate.py) or from within Python (using Python's `help` function or iPython's `?` and `??` shortcuts).

## Analysis

A large number of pre-made scripts for the analysis of camera data, calibration data, and metadata are provided in the [analysis](analysis) subfolder.
These are sorted by the parameter they probe, such as linearity or dark current.
Please refer to the README in the [analysis](analysis) subfolder and documentation in the scripts themselves for further information.
A number of common methods for analysing these data have also been bundled into the [`spectacle.analyse`](spectacle/analyse.py) submodule.

## Calibration

Finally, pre-made scripts for generating calibration data based on data gathered by the user are provided in the [calibration](calibration) subfolder.
These are sorted by the parameter they probe, such as bias or flat-field response.
Furthermore, a script is provided that combines calibration data generated this way into a format that can be uploaded to the [SPECTACLE database](http://spectacle.ddq.nl/).
Please refer to the README in the [calibration](calibration) subfolder and documentation in the scripts themselves for further information.

# Further information

The SPECTACLE method itself has been fully developed and applied, as shown in [our paper](https://doi.org/10.1364/OE.27.019075).
The [SPECTACLE database](http://spectacle.ddq.nl/) and `spectacle` Python module are still in active development.
Contributions from the community are highly welcome and we invite everyone to contribute.

Further information will be added to this repository with time.
If anything is missing, please [raise an issue](https://github.com/monocle-h2020/camera_calibration/issues) or [contact the authors directly](mailto:burggraaff@strw.leidenuniv.nl).
