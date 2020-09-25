# Analysis

This folder contains scripts for analysing camera data and calibration data using the methods described in the SPECTACLE paper (https://doi.org/10.1364/OE.27.019075).
These scripts are used to create plots, tables, etc. which provide insight into the performance and characteristics of the camera.

This README is only intended to give a very broad overview of the results that can be obtained using these scripts.
For further documentation, please refer to that included in the scripts themselves.

## Paper

The following scripts were used to generate the figures from the SPECTACLE paper:
* Fig.  2: [linearity_characterise_multiple.py](linearity_characterise_multiple.py), based on output from [linearity_raw.py](linearity_raw.py) and [linearity_jpeg.py](linearity_jpeg.py)
* Fig.  3: [linearity_plot_response_multiple.py](linearity_plot_response_multiple.py)
* Fig.  4: [jpeg_sRGB_comparison_plot_multiple.py](jpeg_sRGB_comparison_plot_multiple.py), based on output from [jpeg_sRGB_gamma_free.py](jpeg_sRGB_gamma_free.py) and [jpeg_sRGB_gamma_fixed.py](jpeg_sRGB_gamma_fixed.py)
* Fig.  5: [plot_RGBG2_multiple.py](plot_RGBG2_multiple.py)
* Fig.  6: [iso_normalisation_multiple.py](iso_normalisation_multiple.py), based on output from [iso_normalisation.py](../calibration/iso_normalisation.py)
* Fig.  7: [gain_characterise_multiple.py](gain_characterise_multiple.py), based on output from [gain.py](../calibration/gain.py)
* Fig.  8: [gain_characterise_multiple.py](gain_characterise_multiple.py), based on output from [gain.py](../calibration/gain.py)
* Fig.  9: [flatfield_compare_maps.py](flatfield_compare_maps.py), based on output from [flatfield.py](../calibration/flatfield.py)
* Fig. 10: [spectral_response_plot_multiple.py](spectral_response_plot_multiple.py), based on output from [spectral_response_monochromator.py](../calibration/spectral_response_monochromator.py) and [spectral_response_ispex_sun.py](../calibration/spectral_response_ispex_sun.py)
* Fig. 11: This script was generated using a highly hard-coded version of [spectral_response_ispex_sun.py](../calibration/spectral_response_ispex_sun.py). However, it is more easily replicated using the more general [spectral_response_plot_multiple.py](spectral_response_plot_multiple.py) script.

Please note that for the paper, some parameters such as axis labels were manually edited.

## General

[plot_RGBG2_multiple.py](plot_RGBG2_multiple.py) may be used to plot Gaussed maps of any number of data stacks, similar to Fig. 7 in the paper.

## Linearity

Pearson r maps may be generated using [linearity_raw.py](linearity_raw.py) (for RAW data) and [linearity_jpeg.py](linearity_jpeg.py) (for JPEG data).
Histograms of the resulting Pearson r distributions may be plotted using  [linearity_characterise.py](linearity_characterise.py) (for a single camera) or [linearity_characterise_multiple.py](linearity_characterise_multiple.py) (for multiple cameras) to create plots similar to Fig. 2 in the paper.

The response of the central pixels in a camera may be analysed using [linearity_plot_response.py](linearity_plot_response.py) (for a single camera) or [linearity_plot_response_multiple.py](linearity_plot_response_multiple.py) (for multiple cameras) to create plots similar to Fig. 3 in the paper.

## JPEG response

The JPEG response of a camera may be analysed using [jpeg_sRGB_gamma_free.py](jpeg_sRGB_gamma_free.py) (fitting an sRGB response with a free gamma value to each pixel) and [jpeg_sRGB_gamma_fixed.py](jpeg_sRGB_gamma_fixed.py) (comparing them to sRGB models with fixed gamma values).
The results from this may be plotted using [jpeg_sRGB_characterise_model.py](jpeg_sRGB_characterise_model.py) (free gamma), [jpeg_sRGB_comparison_plot.py](jpeg_sRGB_comparison_plot.py) (fixed gamma), or [jpeg_sRGB_comparison_plot_multiple.py](jpeg_sRGB_comparison_plot_multiple.py) (free and fixed gamma, multiple cameras) to create plots similar to Fig. 4 in the paper.

## Bias

Bias maps may be analysed using [bias_characterise.py](bias_characterise.py) to create plots similar to Fig. 5 in the paper.

## Read noise

Read noise maps, in ADU, may be analysed using [readnoise_characterise_ADU.py](readnoise_characterise_ADU.py) to create plots similar to Fig. 5 in the paper.
Another script, [readnoise_characterise_normalised.py](readnoise_characterise_normalised.py), does the same but normalises data according to ISO speed first.

Furthermore, the relation between read noise and ISO speed may be analysed using [readnoise_iso_relation.py](readnoise_iso_relation.py) to create a plot of read noise at each ISO speed.

## Dark current

The dark current in a camera may be analysed using using [dark_characterise_ADU.py](dark_characterise_ADU.py) to create histograms and Gaussed maps of dark current per pixel in normalised ADU/s.
If a gain map has been created, [dark_characterise_electrons.py](dark_characterise_electrons.py) may be used to create similar figures in units of electrons/s.

## ISO speed normalisation

The ISO speed normalisation may be analysed using [iso_normalisation.py](iso_normalisation.py) (single camera) or [iso_normalisation_multiple.py](iso_normalisation_multiple.py) (multiple cameras) to generate plots similar to Fig. 6 in the paper.

## Gain

Gain maps may be analysed using [gain_characterise.py](gain_characterise.py) (single camera) or [gain_characterise_multiple.py](gain_characterise_multiple.py) to create plots similar to Figs. 7 and 8 in the paper.

## Flat-field correction

Flat-field data may be analysed using [flatfield_characterise_data.py](flatfield_characterise_data.py) to create plots similar to Fig. 5 in the paper and plot the SNR of the data. 

Flat-field models may be compared using [flatfield_compare_model_parameters.py](flatfield_compare_model_parameters.py) (comparing model parameters) or [flatfield_compare_maps.py](flatfield_compare_maps.py) (comparing fitted flat-field maps).
The latter may also be used to create plots similar to Fig. 9 in the paper.

## Spectral response

Spectral response curves may be compared at each wavelength using [spectral_response_compare_curves.py](spectral_response_compare_curves.py).

Monochromator data may be analysed using [spectral_response_monochromator_single_spectrum.py](spectral_response_monochromator_single_spectrum.py) (for a spectrum taken with a single setting, e.g. filter/grating) and [spectral_response_monochromator_plot_outputs.py](spectral_response_monochromator_plot_outputs.py) (to plot intermediary results from the calibration process).	

Spectral response curves may be plotted using [spectral_response_plot_multiple.py](spectral_response_plot_multiple.py) to create plots similar to Fig. 10 in the paper.

The SRFs may be compared to the CIE xyz colour-matching functions, and the camera's colour space may be compared to that of the human eye and sRGB in xy space using [xyz_matrix_plot.py](xyz_matrix_plot.py).
To compare multiple cameras to each other, one can use [xyz_matrix_plot_multiple.py](xyz_matrix_plot_multiple.py).
