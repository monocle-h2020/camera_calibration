This folder contains scripts for analysing camera data and calibration data using the methods described in the SPECTACLE paper (https://doi.org/10.1364/OE.27.019075).

Documentation is included in each script, in the form of a header explaining the purpose of the script and how to call it, as well as line-by-line documentation of the code itself.

The following scripts were used to generate the figures from the SPECTACLE paper:
* Fig.  2: [linearity_characterise_multiple.py](linearity_characterise_multiple.py), based on output from [linearity_raw.py](../calibration/linearity_raw.py) and [linearity_jpeg.py](linearity_jpeg.py)
* Fig.  3: [linearity_plot_response_multiple.py](linearity_plot_response_multiple.py)
* Fig.  4: [jpeg_sRGB_comparison_plot_multiple.py](jpeg_sRGB_comparison_plot_multiple.py), based on output from [jpeg_sRGB_gamma_free.py](jpeg_sRGB_gamma_free.py) and [jpeg_sRGB_gamma_fixed.py](jpeg_sRGB_gamma_fixed.py)
* Fig.  5: [plot_RGBG2_multiple.py](plot_RGBG2_multiple.py)
* Fig.  6: [iso_normalisation_multiple.py](iso_normalisation_multiple.py), based on output from [iso_normalisation.py](../calibration/iso_normalisation.py)
* Fig.  7: [gain_characterise_multiple.py](gain_characterise_multiple.py), based on output from [gain.py](../calibration/gain.py)
* Fig.  8: [gain_characterise_multiple.py](gain_characterise_multiple.py), based on output from [gain.py](../calibration/gain.py)
* Fig.  9: [flatfield_compare_maps.py](flatfield_compare_maps.py), based on output from [flatfield.py](../calibration/flatfield.py)
* Fig. 10: [spectral_response_plot_multiple.py](spectral_response_plot_multiple.py), based on output from [spectral_response_monochromator.py](../calibration/spectral_response_monochromator.py) and [spectral_response_ispex_sun.py](../calibration/spectral_response_ispex_sun.py)
* Fig. 11: 
