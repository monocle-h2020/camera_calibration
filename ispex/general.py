from scipy.ndimage.filters import gaussian_filter1d as gauss
from astropy.modeling.blackbody import blackbody_lambda
import numpy as np

y_thick = (1530, 1911)
y_thin  = (1970, 2315)
y = np.concatenate((np.arange(*y_thick), np.arange(*y_thin)))
x_spectrum = (2150, 3900)
x = np.arange(*x_spectrum)

def gauss_filter(D, sigma=5, *args, **kwargs):
    """
    Apply a 1-D Gaussian kernel along one axis
    """
    return gauss(D.astype(float), sigma, *args, axis=1, **kwargs)

def split_spectrum(data):
    thick = data[x_spectrum[0]:x_spectrum[1], y_thick[0]:y_thick[1]]
    thin  = data[x_spectrum[0]:x_spectrum[1], y_thin[0] :y_thin[1] ]
    return thick, thin

def find_white_balance(data):
    return data[x_spectrum[0]:x_spectrum[1], :y_thick[0]-100].mean(axis=(0,1))

def correct_white_balance(data, white_balance):
    return data/white_balance

def blackbody(wavelengths, temperature=5777, norm=1):
    bb = blackbody_lambda(wavelengths*10, temperature).value
    bb = bb / bb.max() * norm
    return bb