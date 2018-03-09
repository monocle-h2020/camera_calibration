from scipy.ndimage.filters import gaussian_filter1d as gauss

def gauss_filter(D, sigma=7, *args, **kwargs):
    """
    Apply a 1-D Gaussian kernel along one axis
    """
    return gauss(D.astype(float), sigma, *args, axis=0, **kwargs)