from scipy.ndimage.filters import gaussian_filter1d as gauss1d, gaussian_filter as gaussMd
from scipy.optimize import curve_fit
from astropy.modeling.blackbody import blackbody_lambda
import numpy as np

y_thick = (1530, 1911)
y_thin  = (1970, 2315)
y = np.concatenate((np.arange(*y_thick), np.arange(*y_thin)))
x_spectrum = (2150, 3900)
x = np.arange(*x_spectrum)

def gauss_filter(D, sigma=5, **kwargs):
    """
    Apply a 1-D Gaussian kernel along one axis
    """
    return gauss1d(D.astype(float), sigma, axis=1, **kwargs)

def gauss_nan(D, sigma=5, **kwargs):
    """
    Apply a multidimensional Gaussian kernel
    https://stackoverflow.com/a/36307291/2229219
    """
    V = D.copy()
    V[D!=D] = 0
    VV = gaussMd(V, sigma=sigma, **kwargs)

    W = 0 * D.copy() + 1
    W[D!=D] = 0
    WW = gaussMd(W, sigma=sigma, **kwargs)

    Z=VV/WW
    return Z

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

def cut(arr, x=250, y=250):
    return arr[y:-y, x:-x]

def bin_centers(bin_left_edges):
    width = bin_left_edges[1] - bin_left_edges[0]
    return bin_left_edges[:-1] + width/2.

def weighted_mean(data, weights, **kwargs):
    """
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
    SS_res = np.ma.sum((y - y_fit)**2, **kwargs)
    SS_tot = np.ma.sum((y - y.mean(**kwargs))**2, **kwargs)
    R2 = 1 - SS_res/SS_tot
    return R2

def RMS(x, **kwargs):
    return np.sqrt(np.mean(x**2, **kwargs))

def distances_px(array):
    x = np.arange(array.shape[1])
    y = np.arange(array.shape[0])
    X, Y = np.meshgrid(x, y)
    x_center, y_center = array.shape[1]/2, array.shape[0]/2
    distance = np.sqrt((X - x_center)**2 + (Y - y_center)**2)
    return X, Y, distance
