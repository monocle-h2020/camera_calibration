import warnings

import numpy as np
from scipy.ndimage.filters import gaussian_filter as gaussMd
from scipy.ndimage.filters import gaussian_filter1d as gauss1d
from scipy.optimize import curve_fit


def gauss_filter(D, sigma=5, **kwargs):
    """
    Apply a 1-D Gaussian kernel along one axis.
    """
    return gauss1d(D.astype(float), sigma, axis=1, **kwargs)


def _gauss_nan(D, sigma=5, **kwargs):
    """
    Apply a multidimensional Gaussian kernel, accounting for NaN values.
    Reference: https://stackoverflow.com/a/36307291/2229219
    """
    V = D.copy()
    V[D!=D] = 0
    VV = gaussMd(V, sigma=sigma, **kwargs)

    W = 0 * D.copy() + 1
    W[D!=D] = 0
    WW = gaussMd(W, sigma=sigma, **kwargs)

    Z=VV/WW
    return Z


def gauss_filter_multidimensional(data, sigma=5, **kwargs):
    """
    Apply a multidimensional Gaussian kernel, accounting for NaN values
    if necessary.

    Select `gaussMd` or `_gauss_nan` depending on if NaN data are present
    in the given `data_element`.
    """
    func = _gauss_nan if np.isnan(data).any() else gaussMd
    data_gauss = func(data, sigma=sigma, **kwargs)

    return data_gauss


def blackbody(wavelengths, temperature=5777, norm=1):
    """
    Return a blackbody curve for `wavelengths` with a given `temperature`
    (default: 5777, effective temperature of the Sun), normalised to a maximum
    value of `norm`.
    """
    h = 6.62607004e-34 # m**2 kg s**-1
    c = 299792458. # m s**-1
    kB = 1.38064852e-23 # m**2 kg s**-2 K**-1
    wavelengths_m = wavelengths/1e9
    bb = (2 * h * c**2) / wavelengths_m**5 * 1/(np.exp((h*c)/(wavelengths_m * kB * temperature)) - 1)
    bb = bb / bb.max() * norm
    return bb


def cut(arr, x=250, y=250):
    """
    Cut the outer `x,y` elements off an array. The outer `y` elements are
    removed from the 0th axis, the outer `x` elements from the 1st axis.
    (corresponding to image data)
    """
    return arr[y:-y, x:-x]


def bin_centers(bin_left_edges):
    """
    Given left edges of a set of bins, return the bin centers
    """
    width = bin_left_edges[1] - bin_left_edges[0]
    return bin_left_edges[:-1] + width/2.


def weighted_mean(data, weights, **kwargs):
    """
    Calculate the weighted mean of `data` with `weights`. Any **kwargs are
    passed to numpy.average.
    Also calculates the weighted sample variance according to
        https://en.wikipedia.org/wiki/Weighted_arithmetic_mean#Weighted_sample_variance
    """
    if len(data) == 1 or len(weights) == 1:
        return data, 1/np.sqrt(weights)
    data = np.array(data) ; weights = np.array(weights)
    mean = np.average(data, weights=weights, **kwargs)
    V1 = np.sum(weights)
    V2 = np.sum(weights**2)
    s2 = np.sum(weights * (data - mean)**2, **kwargs) / (V1 - V2/V1)
    return mean, np.sqrt(s2)


def Rsquare(y, y_fit, **kwargs):
    """
    Calculate the R^2 value of fitted data `y_fit` compared to observations
    `y`. Any additional **kwargs are passed to `numpy.ma.sum`. This can be used
    to calculate R^2 values along an axis `(axis=0)`, for example.
    """
    SS_res = np.ma.sum((y - y_fit)**2, **kwargs)
    SS_tot = np.ma.sum((y - y.mean(**kwargs))**2, **kwargs)
    R2 = 1 - SS_res/SS_tot
    return R2


def RMS(x, **kwargs):
    """
    Calculate the root mean square (RMS) value of an array `x`.
    Any additional **kwargs are passed to `numpy.mean`. This can be used to
    calculate RMS values along an axis `(axis=0)`, for example.
    """
    return np.sqrt(np.mean(x**2, **kwargs))


def uncertainty_from_covariance(covariance):
    """
    Calculate a naive uncertainty estimate from a covariance matrix.
    This is just the square root of the diagonal.
    """
    return np.sqrt(np.diag(covariance))


def correlation_from_covariance(covariance):
    """
    Convert a covariance matrix into a correlation matrix
    https://gist.github.com/wiso/ce2a9919ded228838703c1c7c7dad13b
    """
    v = uncertainty_from_covariance(covariance)
    outer_v = np.outer(v, v)
    correlation = covariance / outer_v
    correlation[covariance == 0] = 0
    return correlation


def generate_XY(shape):
    """
    Given a `shape`, generate a meshgrid of index values in both directions as
    well as a combination.
    """
    x = np.arange(shape[1])
    y = np.arange(shape[0])
    X, Y = np.meshgrid(x, y)
    XY = np.stack([X.ravel(), Y.ravel()])
    return X, Y, XY


def distances_px(array):
    """
    Calculate the distance from the center of `array` for each element.
    """
    X, Y, _ = generate_XY(array.shape)
    x_center, y_center = array.shape[1]/2, array.shape[0]/2
    distance = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
    return X, Y, distance


def symmetric_percentiles(data, percent=0.1, **kwargs):
    """
    Find the lowest and highest `percent` percentile in a data set `data`.
    Default: P_0.1 and P_99.9

    Additional **kwargs are passed to `numpy.nanpercentile`
    """
    return np.nanpercentile(data, percent, **kwargs), np.nanpercentile(data, 100-percent, **kwargs)


def return_with_filename(to_return, filename, return_filename=False):
    """
    Handle optional returns for filenames.

    Inputs:
        to_return: object that must always be returned
        filename: filename that is returned only if return_filename is True
        return_filename: boolean determining in the filename is returned
    """
    if return_filename:
        return to_return, filename
    else:
        return to_return


def deprecation(message):
    warnings.warn(message, DeprecationWarning, stacklevel=2)


def find_matching_file(folder, filename):
    """
    In a given `folder`, find files that end with the `filename`,
    e.g. iPhone_SE_bias.npy for filename="bias.npy".

    This is here instead of in spectacle.io to prevent a circular import between
    io and camera.
    """
    pattern = f"*{filename}"

    # Find all files in `folder` that match the pattern "*{filename}"
    all_files = list(folder.glob(pattern))

    # Check the length of the list and raise an error if it is not 1.
    if len(all_files) == 0:
        raise FileNotFoundError(f"No files matching the pattern `{pattern}` found in `{folder}`")
    elif len(all_files) > 1:
        raise OSError(f"Multiple ({len(all_files)}) files matching the pattern `{pattern}` found in `{folder}`")

    return all_files[0]
