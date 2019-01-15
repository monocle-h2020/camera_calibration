import numpy as np
from .general import Rsquare


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
            print(f"{i/y.shape[1]*100:.1f}%")

    return R2, saturated
