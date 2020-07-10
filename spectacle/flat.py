"""
Code relating to flat-fielding, such as fitting or applying a vignetting model.
"""

import numpy as np
from .general import gaussMd, curve_fit, generate_XY, return_with_filename
from . import raw

parameter_labels = ["k0", "k1", "k2", "k3", "k4", "cx", "cy"]
parameter_error_labels = ["k0_err", "k1_err", "k2_err", "k3_err", "k4_err", "cx_err", "cy_err"]

_clip_border = np.s_[250:-250, 250:-250]


def clip_data(data, borders=_clip_border):
    """
    Make data outside the `borders` NaN, to remove artefacts from mechanical
    vignetting.

    To do:
        * Use camera-dependent default borders.
    """
    # Create an empty array
    data_with_nan = np.tile(np.nan, data.shape)

    # Add the data within the borders to the empty array
    data_with_nan[borders] = data[borders]

    return data_with_nan


def vignette_radial(shape, XY, k0, k1, k2, k3, k4, cx_hat, cy_hat):
    """
    Vignetting function as defined in Adobe DNG standard 1.4.0.0
    Reference:
        https://www.adobe.com/content/dam/acom/en/products/photoshop/pdfs/dng_spec_1.4.0.0.pdf

    Adapted to use a given image shape for conversion to relative coordinates,
    rather than deriving this from the inputs XY.

    Parameters
    ----------
    XY
        array with X and Y positions of pixels, in absolute (pixel) units
    k0, ..., k4
        polynomial coefficients
    cx_hat, cy_hat
        optical center of image, in normalized euclidean units (0-1)
        relative to the top left corner of the image
    """
    x, y = XY

    x0, y0 = 0, 0 # top left corner
    x1, y1 = shape[1], shape[0]  # bottom right corner
    cx = x0 + cx_hat * (x1 - x0)
    cy = y0 + cy_hat * (y1 - y0)
    # (cx, cy) is the optical center in absolute (pixel) units
    mx = max([abs(x0 - cx), abs(x1 - cx)])
    my = max([abs(y0 - cy), abs(y1 - cy)])
    m = np.sqrt(mx**2 + my**2)
    # m is the euclidean distance from the optical center to the farthest corner in absolute (pixel) units
    r = 1/m * np.sqrt((x - cx)**2 + (y - cy)**2)
    # r is the normalized euclidean distance of every pixel from the optical center (0-1)

    p = [k4, 0, k3, 0, k2, 0, k1, 0, k0, 0, 1]
    g = np.polyval(p, r)
    # g is the normalization factor to multiply measured values with

    return g


def fit_vignette_radial(correction_observed, **kwargs):
    """
    Fit a radial vignetting function to the observed correction factors
    `correction_observed`. Any additional **kwargs are passed to `curve_fit`.
    """
    # Coordinates for each pixel
    X, Y, XY = generate_XY(correction_observed.shape)

    # Flatten the data
    correction_flattened = correction_observed.ravel()

    # Find non-NaN elements
    indices_not_nan = np.where(~np.isnan(correction_flattened))[0]
    XY = XY[:, indices_not_nan]
    correction_flattened = correction_flattened[indices_not_nan]

    # Radial vignetting function with fixed shape, so this is not fitted
    vignette_radial_fixed_shape = lambda XY, *parameters: vignette_radial(correction_observed.shape, XY, *parameters)

    # Fit a vignette profile
    popt, pcov = curve_fit(vignette_radial_fixed_shape, XY, correction_flattened, p0=[1, 2, -5, 5, -2, 0.5, 0.5], **kwargs)
    standard_errors = np.sqrt(np.diag(pcov))

    return popt, standard_errors


def apply_vignette_radial(shape, parameters):
    """
    Apply a radial vignetting function to obtain a correction factor map.
    """
    X, Y, XY = generate_XY(shape)
    correction = vignette_radial(shape, XY, *parameters).reshape(shape)
    return correction


def load_flatfield_correction(root, shape, return_filename=False):
    """
    Load the flat-field correction model, the parameters of which are contained
    in `root`/calibration/flatfield_parameters.csv
    """
    filename = root/"calibration/flatfield_parameters.csv"
    data = np.loadtxt(filename, delimiter=",")
    parameters, errors = data[:7], data[7:]
    correction_map = apply_vignette_radial(shape, parameters)

    return return_with_filename(correction_map, filename, return_filename)


def normalise_RGBG2(mean, stds, bayer_pattern):
    """
    Normalise the Bayer RGBG2 channels to 1.
    """
    # Demosaick the data
    mean_RGBG = raw.demosaick(bayer_pattern, mean)
    stds_RGBG = raw.demosaick(bayer_pattern, stds)

    # Convolve with a Gaussian kernel to find the maxima without being
    # sensitive to outliers
    mean_RGBG_gauss = gaussMd(mean_RGBG, sigma=(0,5,5))

    # Find the maximum per channel and cast these into an array of the same
    # shape as the data
    normalisation_factors = mean_RGBG_gauss.max(axis=(1,2))
    normalisation_array = normalisation_factors[:,np.newaxis,np.newaxis]

    # Normalise the mean and standard deviation data to 1
    mean_RGBG = mean_RGBG / normalisation_array
    stds_RGBG = stds_RGBG / normalisation_array

    # Re-mosaick the now-normalised flat-field data
    mean_remosaicked = raw.put_together_from_colours(mean_RGBG, bayer_pattern)
    stds_remosaicked = raw.put_together_from_colours(stds_RGBG, bayer_pattern)

    return mean_remosaicked, stds_remosaicked


def correct_flatfield_from_map(flatfield, data, clip=False):
    """
    Apply a flat-field correction from a flat-field map `flatfield` to an
    array `data`.

    If `clip`, clip the data (make the outer borders NaN).
    """
    if clip:
        data_to_correct = clip_data(data)
    else:
        data_to_correct = data

    # Correct the data
    data_corrected = data_to_correct * flatfield

    return data_corrected
