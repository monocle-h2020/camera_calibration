# Calibration

This folder contains scripts for obtaining calibration data using the methods described in the SPECTACLE paper (https://doi.org/10.1364/OE.27.019075). Results from these scripts, such as look-up tables and maps, are then used to calibrate data in other applications.

This README file contains a brief description of each calibration script. For further documentation, please refer to that included in the scripts themselves.

## General

Most of these scripts require the use of image stacks, generated using the [stack_mean_std.py](../tools/stack_mean_std.py) script, rather than individual RAW images. This is either because they are based on statistical properties of such image stacks (e.g. for read noise) or because they are highly noise-sensitive (e.g. for gain).

## Metadata

Some metadata are necessary for the calibration and analysis of camera data, such as the pattern of the Bayer RGBG2 channels. These metadata files can be retrieved from the [SPECTACLE database](http://spectacle.ddq.nl/) or generated using the [generate_metadata.py](generate_metadata.py) script. This script is used to generate a metadata file based on a single saturated image and some user inputs.

## Bias

A bias map is generated using [bias.py](bias.py). This map has the mean bias value per pixel at the lowest ISO speed given. This is corrected using the `spectacle` function `spectacle.calibrate.correct_bias`.

The bias map depends on ISO speed and is unique to each camera. Even two cameras of the same model do not share a bias map. For this reason, a unique bias map must be generated for each device to take full advantage of the bias correction. However, for many purposes this is not necessary as the inter-pixel differences in bias are often relatively minor. In this case, a mean value may be used, either derived from measurement or given in metadata.

`readnoise.py` is used to generate a read noise map (read noise in each pixel). This is not used in the image calibration process, but can be used to characterise the noise response of a camera.

`iso_normalisation.py` is used to generate a look-up table for normalising data taken at different ISO speeds. This is used to normalise data using the `spectacle` function `spectacle.calibrate.normalise_iso`.

`dark_current.py` is used to generate a dark current map (dark current in each pixel). This can be corrected using the `spectacle` function `spectacle.calibrate.correct_dark_current`.

`gain.py` is used to generate a gain map (gain in ADU per electron in each pixel). This is not used in the further calibration but can be used to analyse, for example, dark current and read noise in terms of electrons instead of ADU.

`flatfield.py` is used to generate a look-up table and model for the flat-field response (sensitivity) per pixel. This can be corrected using the `spectacle` function `spectacle.calibrate.correct_flatfield`.

## Spectral response

Two methods for characterising the spectral response of a camera are currently included in MONOCLE. These are the use of a monochromator and the use of a spectrometer such as iSPEX.

In either case, the spectral response curves are saved to `root/calibration/spectral_response.npy`. The `spectacle` method `spectacle.calibration.correct_spectral_response` may then be used to correct spectral data (e.g. from iSPEX) by dividing out the spectral response of the camera. Further calibration methods, for example to convert simple RGB images to the CIE XYZ colour space, are currently in development on the [xyz branch](https://github.com/monocle-h2020/camera_calibration/tree/xyz).

### Monochromator

Monochromator data take the form of image stacks taken at specific wavelengths. Multiple data sets may be combined, for example taken with different gratings or filters.

The [spectral_response_monochromator.py](spectral_response_monochromator.py) script is used to process monochromator data and generate a spectral response curve. This script is currently somewhat hard-coded for data from the NERC Field Spectroscopy Facility (as described in the SPECTACLE paper), but it should be simple to add functionality for data from other instruments.

### Spectrometry

Spectrometric data take the form of images containing a full spectrum. These can be generated for example with iSPEX, a smartphone spectrometer add-on, as described in the SPECTACLE paper.

The [spectral_response_ispex_sun.py](spectral_response_ispex_sun.py) script is used to process iSPEX data and generate a spectral response curve. However, this script is not very portable, as such measurements often differ for example in the location of the script on the camera (which is device-dependent). Furthermore, it contains some hard-coded assumptions on iSPEX which may not apply to other spectrometric set-ups. For this reason we recommend using this script as an inspiration for writing your own, rather than blindly running it on your own data.

As part of the [H2020 consortium MONOCLE](https://monocle-h2020.eu/Home), the authors of SPECTACLE are currently developing a new, universal version of iSPEX (in fact, this is what inspired the development of SPECTACLE itself). This will come with a more extensive data processing library specifically for iSPEX data, which may replace this functionality.
