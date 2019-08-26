This folder contains scripts for obtaining calibration data using the methods described in the SPECTACLE paper (https://doi.org/10.1364/OE.27.019075).

`generate_metadata.py` is used to generate a metadata file based on a single saturated image and some user inputs. This metadata file contains information, such as the Bayer colour pattern of the camera, that is necessary for all further calibration and analysis steps.

`linearity_raw.py` is used to determine the linearity per pixel in RAW data (expressed through the Pearson r coefficient). This is not used in the image calibration process, but can be used to flag bad pixels. A similar script for JPEG data is available at [analysis/linearity_jpeg.py](../analysis/linearity_jpeg.py)

`bias.py` is used to generate a bias map (bias offset in each pixel). This can be corrected using the `spectacle` function `spectacle.calibrate.correct_bias`.

`readnoise.py` is used to generate a read noise map (read noise in each pixel). This is not used in the image calibration process, but can be used to characterise the noise response of a camera.

`iso_normalisation.py` is used to generate a look-up table for normalising data taken at different ISO speeds. This is used to normalise data using the `spectacle` function `spectacle.calibrate.normalise_iso`.

`dark_current.py` is used to generate a dark current map (dark current in each pixel). This can be corrected using the `spectacle` function `spectacle.calibrate.correct_dark_current`.

`gain.py` is used to generate a gain map (gain in ADU per electron in each pixel). This is not used in the further calibration but can be used to analyse, for example, dark current and read noise in terms of electrons instead of ADU.

`flatfield.py` is used to generate a look-up table and model for the flat-field response (sensitivity) per pixel. This can be corrected using the `spectacle` function `spectacle.calibrate.correct_flatfield`.

TO DO: spectral response.
