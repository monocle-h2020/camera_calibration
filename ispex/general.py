from scipy.ndimage.filters import gaussian_filter1d as gauss

def gauss_filter(D, sigma=7, *args, **kwargs):
    """
    Apply a 1-D Gaussian kernel along one axis
    """
    return gauss(D.astype(float), sigma, *args, axis=0, **kwargs)

def white_balance(data, x_spectrum, y_spectrum):
    return data[x_spectrum[0]:x_spectrum[1], :y_spectrum-100].mean(axis=(0,1))