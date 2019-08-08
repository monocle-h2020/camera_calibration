import numpy as np
from .general import Rsquare, curve_fit, RMS
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


def sRGB(I):
    u = I/255.
    u_small = np.where(u < 0.0031308)
    u_large = np.where(u >=0.0031308)
    u[u_small] = 12.92 * u[u_small]
    u[u_large] = 1.055 * u[u_large]**(1/2.4) - 0.055
    u *= 255
    return u


def sRGB_generic(I, normalization=255, gamma=2.4):
    u = I/normalization
    u_small = np.where(u < 0.0031308)
    u_large = np.where(u >=0.0031308)
    u[u_small] = 12.92 * u[u_small]
    u[u_large] = 1.055 * u[u_large]**(1/gamma) - 0.055
    u *= 255
    u = np.clip(u, 0, 255)
    return u


def fit_sRGB_generic(intensities, jmeans):
    normalizations = np.tile(np.nan, jmeans.shape[1:])
    gammas = normalizations.copy()
    Rsquares = normalizations.copy()
    try:
        for i in range(jmeans.shape[1]):
            for j in range(jmeans.shape[2]):
                for k in range(jmeans.shape[3]):
                    popt, pcov = curve_fit(sRGB_generic, intensities, jmeans[:,i,j,k], p0=[1, 2.2])
                    normalizations[i,j,k], gammas[i,j,k] = popt
                    jmeans_fit = sRGB_generic(intensities, *popt)
                    Rsquares[i,j,k] = Rsquare(jmeans[:,i,j,k], jmeans_fit)
            if i%10 == 0:
                print(100*i/jmeans.shape[1])
    except BaseException as e:  # BaseException so we also catch SystemExit and KeyboardInterrupt
        print(e)
        pass
    finally:
        return normalizations, gammas, Rsquares

def sRGB_compare_gamma(intensities, jmeans, gamma):
    normalizations = np.tile(np.nan, jmeans.shape[1:])
    Rsquares = normalizations.copy()
    RMSes = normalizations.copy()
    RMSes_relative = normalizations.copy()
    sRGB = lambda I, normalization: sRGB_generic(I, normalization, gamma=gamma)
    try:
        for i in range(jmeans.shape[1]):
            for j in range(jmeans.shape[2]):
                for k in range(jmeans.shape[3]):
                    popt, pcov = curve_fit(sRGB, intensities, jmeans[:,i,j,k], p0=[1])
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
