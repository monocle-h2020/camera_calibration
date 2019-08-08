import numpy as np

def effective_bandwidth(wavelengths, response, axis=0, **kwargs):
    response_normalised = response / response.max(axis=axis)
    return np.trapz(response_normalised, x=wavelengths, axis=axis, **kwargs)


wavelengths_interpolated = np.arange(390, 701, 1)


def interpolate(wavelengths, response, interpolate_to=wavelengths_interpolated):
    interpolated = np.stack([np.interp(interpolate_to, wavelengths, R) for R in response.T]).T
    return interpolate_to, interpolated