import numpy as np

def effective_bandwidth(wavelengths, response, axis=0, **kwargs):
    response_normalised = response / response.max(axis=axis)
    return np.trapz(response_normalised, x=wavelengths, axis=axis, **kwargs)
