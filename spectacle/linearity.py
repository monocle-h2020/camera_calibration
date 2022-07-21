import numpy as np
from scipy.stats import pearsonr
from functools import partial

from .general import Rsquare, curve_fit, RMS
from . import io

# minimum Pearson r value to be considered linear (see SPECTACLE paper)
linearity_limit = 0.98
modes = {"p": "polarisers", "t": "exposure_time"}


def calibration_mode(mode_abbreviated):
    """
    Given a command-line input, return a string describing the relevant
    calibration mode.
    """
    try:
        mode_full = modes[mode_abbreviated]
    except KeyError:
        raise ValueError(f"Invalid linearity calibration mode `{mode_abbreviated}` given; must be one of `{modes.keys()}`")
    else:
        return mode_full


def malus(angle, offset=0):
    """
    Use Malus's law to calculate the transmission of two linear polarisers held
    at `angle` relative to each other. An `offset` can be added (default 0).

    `angle`, `offset` should be in degrees.
    """
    return (np.cos(np.radians(angle-offset)))**2


def malus_error(angle0, angle1=0, I0=1., sigma_angle0=2., sigma_angle1=0.1, sigma_I0=0.01):
    """
    Calculate the error in the transmission of two linear polarisers held at
    `angle0` and `angle1` respectively, using Malus's law. If only `angle0` is
    given, it is assumed to be the angle between the two polarisers.
    `angle0` and `angle1` have associated errors `sigma_angle0` and
    `sigma_angle1`.

    `I0` is a normalisation factor for the incoming intensity, with associated
    error `sigma_I0`.

    `angle0`, `angle1`, `sigma_angle0`, `sigma_angle1` should be in degrees.
    """
    alpha = angle0 - angle1
    A = I0 * np.pi/180 * np.sin(np.pi/90 * (alpha))
    s_a2 = A**2 * (sigma_angle0**2 + sigma_angle1**2)
    s_I2 = (malus(angle0, offset=angle1) * sigma_I0)**2
    total = np.sqrt(s_I2 + s_a2)

    return total


def sRGB(I, normalization=255, gamma=2.4):
    """
    Apply an sRGB-like response to an intensity `I`. The `normalization` and
    `gamma` are parameters.
    """
    u = I/normalization
    u_small = np.where(u < 0.0031308)
    u_large = np.where(u >=0.0031308)
    u[u_small] = 12.92 * u[u_small]
    u[u_large] = 1.055 * u[u_large]**(1/gamma) - 0.055
    u *= 255.
    u = np.clip(u, 0, 255)
    return u


def sRGB_inverse(I, normalization=255, gamma=2.4):
    """
    Apply an inverse sRGB-like response to data `I`. The `normalization` and
    `gamma` are parameters.
    """
    u = I/255.
    u_small = np.where(u <= 0.04045)
    u_large = np.where(u >  0.04045)
    u[u_small] = u[u_small]/12.92
    u[u_large] = ((u[u_large]+0.055)/ 1.055)**gamma
    u *= normalization
    return u


def fit_sRGB_generic(intensities, jmeans):
    """
    Fit a generic sRGB profile (normalization and gamma as free parameters) to
    `intensities` and responses `jmeans`.

    Returns the best-fitting normalization and gamma, as well as the respective
    R^2 for this fit, for each pixel.
    """
    normalizations = np.full(jmeans.shape[1:], np.nan)
    gammas = normalizations.copy()
    Rsquares = normalizations.copy()
    try:
        for i in range(jmeans.shape[1]):
            for j in range(jmeans.shape[2]):
                for k in range(jmeans.shape[3]):
                    popt, pcov = curve_fit(sRGB, intensities, jmeans[:,i,j,k], p0=[1, 2.2])
                    normalizations[i,j,k], gammas[i,j,k] = popt
                    jmeans_fit = sRGB(intensities, *popt)
                    Rsquares[i,j,k] = Rsquare(jmeans[:,i,j,k], jmeans_fit)
            if i%10 == 0:
                print(100*i/jmeans.shape[1])
    except BaseException as e:  # BaseException so we also catch SystemExit and KeyboardInterrupt
        print(e)
        pass
    finally:
        return normalizations, gammas, Rsquares


def sRGB_compare_gamma(intensities, jmeans, gamma):
    """
    Fit sRGB profiles with a given `gamma` and free normalization to given
    `intensities` and responsese `jmeans`. Calculate the RMS difference between
    the best-fitting model with `gamma` and the data.
    """
    normalizations = np.full(jmeans.shape[1:], np.nan)
    Rsquares = normalizations.copy()
    RMSes = normalizations.copy()
    RMSes_relative = normalizations.copy()
    sRGB_here = partial(sRGB, gamma=gamma)
    try:
        for i in range(jmeans.shape[1]):
            for j in range(jmeans.shape[2]):
                for k in range(jmeans.shape[3]):
                    popt, pcov = curve_fit(sRGB_here, intensities, jmeans[:,i,j,k], p0=[1])
                    normalizations[i,j,k] = popt[0]
                    ind = np.where(jmeans[:,i,j,k] < 255)
                    jmeans_fit = sRGB(intensities[ind], *popt)
                    Rsquares[i,j,k] = Rsquare(jmeans[:,i,j,k][ind], jmeans_fit)
                    RMSes[i,j,k] = RMS(jmeans[:,i,j,k][ind] - jmeans_fit)
                    RMSes_relative[i,j,k] = RMS(1 - jmeans[:,i,j,k][ind] / jmeans_fit)
            if i%30 == 0:
                print(100*i/jmeans.shape[1])
    except BaseException:  # BaseException so we also catch SystemExit and KeyboardInterrupt
        pass
    finally:
        return normalizations, Rsquares, RMSes, RMSes_relative


def pearson_r_single(x, y, saturate):
    """
    Calculate the Pearson r correlation between `x` and `y`, ignoring data
    above the saturation limit `saturate`.
    """
    ind = np.where(y < saturate)
    r, p = pearsonr(x[ind], y[ind])
    return r


def calculate_pearson_r_values(x, y, **kwargs):
    """
    Apply 'pearson_r_single' for every pixel in `y` (two-dimensional).

    Use this for RAW data.
    """
    saturated = []
    r = np.zeros(y.shape[1:])
    for i in range(y.shape[1]):
        for j in range(y.shape[2]):
            try:
                r[i,j] = pearson_r_single(x, y[:,i,j], **kwargs)
            except (TypeError, ValueError):  # if fully saturated
                r[i,j] = np.nan
                saturated.append((i,j))
        if i%5 == 0:
            print(f"{i/y.shape[1]*100:.1f}%", end=" ", flush=True)

    return r, saturated


def calculate_pearson_r_values_jpeg(x, y, **kwargs):
    """
    Apply 'pearson_r_single' for every pixel in `y` (three-dimensional).

    Use this for JPEG data.
    """
    r, saturated = [[], [], []], [[], [], []]
    for j in range(3):
        r[j], saturated[j] = calculate_pearson_r_values(x, y[..., j], saturate=240)
    r = np.stack(r)
    return r, saturated


def filename_to_intensity(filename):
    """
    Split filenames according to one of the standards (see below) and convert
    them to intensities.

    Currently supported standards:
        * Polariser angles: io.split_pol_angle
        * Exposure time: io.split_exposure_time
    """
    if "pol" in filename.stem:
        angle = io.split_pol_angle(filename)
        offset_angle = np.loadtxt(filename.parent/"default_angle.dat").ravel()[0]
        intensity = malus(angle, offset_angle)
        intensity_error = malus_error(angle, offset_angle, sigma_angle0=1, sigma_angle1=1)
    elif "t" in filename.stem:
        time = io.split_exposure_time(filename)
        intensity = time
        intensity_error = 0.  # temporary
    else:
        raise ValueError(f"Unknown filename format {filename}")

    return intensity, intensity_error
