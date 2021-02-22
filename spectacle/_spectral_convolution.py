"""
Module with functions for band-averaging
"""

import numpy as np
from scipy.integrate import simps
import warnings


def integrate(*args, **kwargs):
    try:
        result = simps(*args, **kwargs)
    except IndexError:
        result = 0.

    return result


def interpolate_spectral_data(band_wavelengths, data_wavelengths, data_response, extrapolation_value=np.nan):
    data_interpolated = np.interp(band_wavelengths, data_wavelengths, data_response, left=extrapolation_value, right=extrapolation_value)
    return data_interpolated


def check_spectral_overlap(band_wavelengths, band_response, data_wavelengths, threshold=0.05):
    # Shortest and longest wavelength in the data
    left, right = data_wavelengths[0], data_wavelengths[-1]

    # Calculate the full band integrals
    integral_full = integrate(band_response, x=band_wavelengths)
    integral_left = integrate(band_response[band_wavelengths < left], x=band_wavelengths[band_wavelengths < left])
    integral_right = integrate(band_response[band_wavelengths > right], x=band_wavelengths[band_wavelengths > right])
    integral_without_overlap = integral_left + integral_right
    integral_ratio = integral_without_overlap / integral_full

    # If the data wavelengths fall entirely within the band wavelengths, return True
    if integral_without_overlap == 0:
        return True

    # If the area without overlap represents less than `threshold` of the total, return True
    if integral_ratio <= threshold:
        return True
    else:
        return False


def nan_values(data_response):
    return np.tile(np.nan, len(data_response))


def adjust_band_wavelengths(band_wavelengths, band_response, data_wavelengths):
    left, right = data_wavelengths[0], data_wavelengths[-1]
    indices = np.where((band_wavelengths >= left) & (band_wavelengths <= right))
    new_wavelengths = band_wavelengths[indices]
    new_response = band_response[indices]
    return new_wavelengths, new_response


def bandaverage(band_wavelengths, band_response, data_wavelengths, data_response):
    if not check_spectral_overlap(band_wavelengths, band_response, data_wavelengths):
        return nan_values(data_response)
    else:
        band_wavelengths, band_response = adjust_band_wavelengths(band_wavelengths, band_response, data_wavelengths)

    response_interpolated = interpolate_spectral_data(band_wavelengths, data_wavelengths, data_response)
    response_multiplied = response_interpolated * band_response
    response_sum = integrate(response_multiplied, x=band_wavelengths)
    weight_sum = integrate(band_response, x=band_wavelengths)
    response_average = response_sum / weight_sum
    return response_average


def bandaverage_multi(band_wavelengths, band_response, data_wavelengths, data_response_multi):
    if not check_spectral_overlap(band_wavelengths, band_response, data_wavelengths):
        return nan_values(data_response_multi)
    else:
        band_wavelengths, band_response = adjust_band_wavelengths(band_wavelengths, band_response, data_wavelengths)

    response_interpolated = np.array([interpolate_spectral_data(band_wavelengths, data_wavelengths, data_response) for data_response in data_response_multi])
    response_multiplied = response_interpolated * band_response
    response_sum = integrate(response_multiplied, x=band_wavelengths, axis=1)
    weight_sum = integrate(band_response, x=band_wavelengths)
    response_average = response_sum / weight_sum
    return response_average
