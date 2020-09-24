# Calibration

This folder contains scripts for obtaining calibration data using the methods described in the SPECTACLE paper (https://doi.org/10.1364/OE.27.019075).
Results from these scripts, such as look-up tables and maps, are then used to calibrate data in other applications.

This README file contains a brief description of each calibration script.
For further documentation, please refer to that included in the scripts themselves.

## General

Most of these scripts require the use of image stacks, generated using the [stack_mean_std.py](../tools/stack_mean_std.py) script, rather than individual RAW images.
This is either because they are based on statistical properties of such image stacks (e.g. for read noise) or because they are highly noise-sensitive (e.g. for gain).

## Camera information

Some camera information is necessary for the calibration and analysis of camera data, such as the pattern of the Bayer RGBG2 channels.
These information files can be retrieved from the [SPECTACLE database](http://spectacle.ddq.nl/) or generated using the [generate_camera.py](generate_camera.py) script.
This script is used to generate a file based on a single image.
If the image has saturated pixels, the bit depth can be determined automatically; otherwise, user input is required.
User input is also required for the camera name, either to confirm one automatically determined from a filename or to input one manually.

## Camera settings

Additional camera information may be added by the user with the [camera_settings.py](camera_settings.py) script.
This script will prompt the user with several questions on camera settings, and saves the results to file.
The resulting settings file is required for some calibrations, such as ISO speed normalisation.

## Bias

A bias map is generated using [bias.py](bias.py).
This map has the mean bias value per pixel at the lowest ISO speed given.
This is corrected using the `spectacle` function `spectacle.calibrate.correct_bias`.

The bias map depends on ISO speed and is unique to each camera.
Even two cameras of the same model do not share a bias map.
For this reason, a unique bias map must be generated for each device to take full advantage of the bias correction.
However, for many purposes this is not necessary as the inter-pixel differences in bias are often relatively minor.
In this case, a mean value may be used, either derived from measurement or given in metadata.

## Read noise

A read noise map is generated using [readnoise.py](readnoise.py).
This map has the mean read noise per pixel at the lowest ISO speed given.
This is not corrected for in the image calibration process, but can be used in an error budget.

The read noise map depends on ISO speed and is unique to each camera.
Even two cameras of the same model do not share a read noise map.
For this reason, a unique read noise map must be generated for each device.
However, for many purposes this is not necessary as the inter-pixel differences in read noise are often relatively minor.
In this case, a mean value may be used, which is derived from measurements.

## ISO speed normalisation

An ISO speed normalisation look-up table is generated using [iso_normalisation.py](iso_normalisation.py), based on image stacks taken at different ISO speeds.
This is used to normalise data using the `spectacle` function `spectacle.calibrate.normalise_iso`.

The ISO speed normalisation calibration requires a bias correction and camera settings file.
If no measured bias map is available, one is generated from the camera information file.

## Dark current

A dark current map is generated using [dark_current.py](dark_current.py).
This map has the mean dark current per second in each pixel, normalised for ISO speed.
This can be corrected using the `spectacle` function `spectacle.calibrate.correct_dark_current`.

The dark current calibration requires an ISO speed normalisation look-up table.

## Gain

A gain map is generated using [gain.py](gain.py).
This map has the gain (in normalised ADU/photoelectron) in each pixel.
This is not used in further calibration, but can be used to convert a signal to photoelectrons with the `spectacle` function `spectacle.calibrate.convert_to_photoelectrons`.

The gain calibration requires a bias correction.
If no measured bias map is available, one is generated from the camera information file.

The gain calibration requires an ISO speed normalisation look-up table.

## Flat-field

A flat-field model is generated using [flatfield.py](flatfield.py).
This model describes the flat-field response (sensivity) in each pixel with seven parameters, as described in the SPECTACLE paper.
The flat-field response is corrected for using the `spectacle` function `spectacle.calibrate.correct_flatfield`.

The flat-field calibration requires a bias correction.
If no measured bias map is available, one is generated from the camera information file.

## Spectral response

Two methods for characterising the spectral response of a camera are currently included in MONOCLE.
These are the use of a monochromator and the use of a spectrometer such as iSPEX.

In either case, the spectral response curves are saved, in CSV format, to `root/calibration/spectral_response.csv`.
The `spectacle` method `spectacle.calibrate.correct_spectral_response` may then be used to correct spectral data (e.g. from iSPEX) by dividing out the spectral response of the camera.
Further calibration methods, for example to convert simple RGB images to the CIE XYZ colour space, are currently in development on the [xyz branch](https://github.com/monocle-h2020/camera_calibration/tree/xyz).

The effective spectral bandwidths are saved to `root/calibration/spectral_bands.csv` and may be loaded with the `spectacle` method `spectacle.calibrate.load_spectral_bands`.

### Monochromator

Monochromator data take the form of image stacks taken at specific wavelengths.
Multiple data sets may be combined, for example taken with different gratings or filters.

The [spectral_response_monochromator.py](spectral_response_monochromator.py) script is used to process monochromator data and generate a spectral response curve.
This script is currently somewhat hard-coded for data from the NERC Field Spectroscopy Facility (as described in the SPECTACLE paper), but it should be simple to add functionality for data from other instruments.

### Spectrometry

Spectrometric data take the form of images containing a full spectrum.
These can be generated for example with iSPEX, a smartphone spectrometer add-on, as described in the SPECTACLE paper.

The [spectral_response_ispex_sun.py](spectral_response_ispex_sun.py) script is used to process iSPEX data and generate a spectral response curve.
However, this script is not very portable, as such measurements often differ for example in the location of the script on the camera (which is device-dependent).
Furthermore, it contains some hard-coded assumptions on iSPEX which may not apply to other spectrometric set-ups.
For this reason we recommend using this script as an inspiration for writing your own, rather than blindly running it on your own data.

As part of the [H2020 consortium MONOCLE](https://monocle-h2020.eu/Home), the authors of SPECTACLE are currently developing a new, universal version of iSPEX (in fact, this is what inspired the development of SPECTACLE itself).
This will come with a more extensive data processing library specifically for iSPEX data, which may replace this functionality.
