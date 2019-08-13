This folder contains scripts for obtaining calibration data using the methods described in the SPECTACLE paper (https://doi.org/10.1364/OE.27.019075).

`bias.py` is used to generate a bias map (bias offset in each pixel). This can be corrected using the `spectacle` function `spectacle.calibrate.correct_bias`.

`readnoise.py` is used to generate a read noise map (read noise in each pixel). This is not used in the image calibration process, but can be used to characterise the noise response of a camera.

`linearity_raw.py` is used to determine the linearity per pixel in RAW data (expressed through the Pearson r coefficient). This is not used in the image calibration process, but can be used to flag bad pixels.

TO DO: dark current.

`iso_normalisation.py` is used to generate a look-up table for normalising data taken at different ISO speeds. This is used to normalise data using the `spectacle` function `spectacle.calibrate.normalise_iso`.

TO DO: gain.

`flatfield.py` is used to generate a look-up table and model for the flat-field response (sensitivity) per pixel. This can be corrected using the `spectacle` function `spectacle.calibrate.correct_flatfield`.

TO DO: spectral response.
