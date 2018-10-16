import numpy as np
from scipy.optimize import curve_fit
from phonecal.general import Rsquare

polariser_angle = 0

def malus(angle, offset=polariser_angle):
    return (np.cos(np.radians(angle-offset)))**2

def malus_error(angle0, angle1=polariser_angle, I0=1., sigma_angle0=2., sigma_angle1=0.1, sigma_I0=0.01):
    alpha = angle0 - angle1
    A = I0 * np.pi/180 * np.sin(np.pi/90 * (alpha))
    s_a2 = A**2 * (sigma_angle0**2 + sigma_angle1**2)
    s_I2 = (malus(angle0, offset=angle1) * sigma_I0)**2
    total = np.sqrt(s_I2 + s_a2)

    return total

def model_knee(iso, slope, offset, knee):
    results = np.tile(knee * slope + offset, len(iso))
    results[iso < knee] = iso[iso < knee] * slope + offset
    return results

def model_knee_error(iso, popt, pcov):
    results = np.tile(popt[2]**2 * pcov[0,0] + popt[0]**2 * pcov[2,2] + pcov[1,1], len(iso))
    results[iso < popt[2]] = iso[iso < popt[2]]**2 * pcov[0,0] + pcov[1,1]
    results = np.sqrt(results)
    return results

def model_linear(x, *params):
    return np.polyval(params, x)

def model_linear_error(x, params, covariances):
    return np.sqrt(covariances[0,0]**2 * x**2 + covariances[1,1]**2)

def fit_iso_relation(isos, inverse_gains, inverse_gain_errors=None):
    try:
        weights = 1/inverse_gain_errors
    except TypeError:
        weights = None

    params_linear, covariance_linear = np.polyfit(isos, inverse_gains, 1, w=weights, cov=True)
    params_knee, covariance_knee = curve_fit(model_knee, isos, inverse_gains, p0=[0.1, 0.1, 200], sigma=inverse_gain_errors)

    models     = [model_linear      , model_knee      ]
    model_errs = [model_linear_error, model_knee_error]
    parameters = [params_linear     , params_knee     ]
    covariances= [covariance_linear , covariance_knee ]

    R2s = np.array([Rsquare(inverse_gains, model(isos, *params)) for model, params in zip(models, parameters)])
    best = R2s.argmax()

    return models[best], model_errs[best], parameters[best], covariances[best], R2s[best]
