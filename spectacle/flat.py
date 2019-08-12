import numpy as np
from .general import distances_px, curve_fit, generate_XY


clip_border = np.s_[250:-250, 250:-250]


def vignette_radial(XY, k0, k1, k2, k3, k4, cx_hat, cy_hat):
    """
    Vignetting function as defined in Adobe DNG standard 1.4.0.0
    Reference:
        https://www.adobe.com/content/dam/acom/en/products/photoshop/pdfs/dng_spec_1.4.0.0.pdf

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

    x0, y0 = x[0], y[0] # top left corner
    x1, y1 = x[-1], y[-1]  # bottom right corner
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
    X, Y, XY = generate_XY(correction_observed.shape)
    popt, pcov = curve_fit(vignette_radial, XY, correction_observed.ravel(), p0=[1, 2, -5, 5, -2, 0.5, 0.5], **kwargs)
    standard_errors = np.sqrt(np.diag(pcov))
    return popt, standard_errors


def apply_vignette_radial(shape, parameters):
    """
    Apply a radial vignetting function to obtain a correction factotr map.
    """
    X, Y, XY = generate_XY(shape)
    correction = vignette_radial(XY, *parameters).reshape(shape)
    return correction


def read_flat_field_correction(root, shape):
    """
    Load the flat-field correction model, the parameters of which are contained
    in `root`/products/flat_parameters.npy
    """
    parameters, errors = np.load(root/"products/flat_parameters.npy")
    correction_map = apply_vignette_radial(shape, parameters)
    return correction_map
