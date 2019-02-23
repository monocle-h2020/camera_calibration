import numpy as np
from .general import Rsquare
from scipy.stats import pearsonr

polariser_angle = 74
linearity_limit = 0.98


def malus(angle, offset=polariser_angle):
    return (np.cos(np.radians(angle-offset)))**2


def malus_error(angle0, angle1=polariser_angle, I0=1., sigma_angle0=2., sigma_angle1=0.1, sigma_I0=0.01):
    alpha = angle0 - angle1
    A = I0 * np.pi/180 * np.sin(np.pi/90 * (alpha))
    s_a2 = A**2 * (sigma_angle0**2 + sigma_angle1**2)
    s_I2 = (malus(angle0, offset=angle1) * sigma_I0)**2
    total = np.sqrt(s_I2 + s_a2)

    return total


def linear_R2(x, y, saturate=4000):
    ind = np.where(y < saturate)
    p = np.polyfit(x[ind], y[ind], 1)
    pv = np.polyval(p, x[ind])
    R2 = Rsquare(y[ind], pv)
    return R2


def calculate_linear_R2_values(x, y, **kwargs):
    saturated = []
    R2 = np.zeros(y.shape[1:])
    for i in range(y.shape[1]):
        for j in range(y.shape[2]):
            try:
                R2[i,j] = linear_R2(x, y[:,i,j], **kwargs)
            except TypeError:  # if fully saturated
                R2[i,j] = np.nan
                saturated.append((i,j))
        if i%5 == 0:
            print(f"{i/y.shape[1]*100:.1f}%", end=" ", flush=True)

    return R2, saturated


def pearson_r_single(x, y, saturate):
    ind = np.where(y < saturate)
    r, p = pearsonr(x[ind], y[ind])
    return r


def calculate_pearson_r_values(x, y, **kwargs):
    saturated = []
    r = np.zeros(y.shape[1:])
    for i in range(y.shape[1]):
        for j in range(y.shape[2]):
            try:
                r[i,j] = pearson_r_single(x, y[:,i,j], **kwargs)
            except TypeError:  # if fully saturated
                r[i,j] = np.nan
                saturated.append((i,j))
        if i%5 == 0:
            print(f"{i/y.shape[1]*100:.1f}%", end=" ", flush=True)

    return r, saturated


def calculate_pearson_r_values_jpeg(x, y, **kwargs):
    r, saturated = [[], [], []], [[], [], []]
    for j in range(3):
        r[j], saturated[j] = calculate_pearson_r_values(x, y[..., j], saturate=240)
    r = np.stack(r)
    return r, saturated


def percentile_r(data):
    ravel = data.ravel()
    return np.percentile(ravel, 0.1), np.percentile(ravel, 99.9)
