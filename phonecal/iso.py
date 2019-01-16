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

def model_knee_label(params, covariances):
    label_model = f"slope: {params[0]:.4f}\noffset: {params[1]:.4f}\nknee: {params[2]:.1f}"
    label_error = f"$\sigma$ slope: {np.sqrt(covariances[0,0]):.4f}\n$\sigma$ offset: {np.sqrt(covariances[1,1]):.4f}\n$\sigma$ knee: {np.sqrt(covariances[2,2]):.1f}"
    return label_model, label_error

def model_linear(x, *params):
    return np.polyval(params, x)

def model_linear_error(x, params, covariances):
    return np.sqrt(covariances[0,0]**2 * x**2 + covariances[1,1]**2)

def model_linear_label(params, covariances):
    label_model = f"slope: {params[0]:.4f}\noffset: {params[1]:.4f}"
    label_error = f"$\sigma$ slope: {np.sqrt(covariances[0,0]):.4f}\n$\sigma$ offset: {np.sqrt(covariances[1,1]):.4f}"
    return label_model, label_error

def fit_iso_relation(isos, inverse_gains, inverse_gain_errors=None):
    try:
        weights = 1/inverse_gain_errors
    except TypeError:
        weights = None

    try:
        params_linear, covariance_linear = np.polyfit(isos, inverse_gains, 1, w=weights, cov=True)
    except ValueError:
        params_linear = np.polyfit(isos, inverse_gains, 1, w=weights, cov=False)
        covariance_linear = np.array([[np.nan, np.nan], [np.nan, np.nan]])
    params_knee, covariance_knee = curve_fit(model_knee, isos, inverse_gains, p0=[0.1, 0.1, 200], sigma=inverse_gain_errors)

    models     = [model_linear      , model_knee      ]
    model_errs = [model_linear_error, model_knee_error]
    model_label= [model_linear_label, model_knee_label]
    parameters = [params_linear     , params_knee     ]
    covariances= [covariance_linear , covariance_knee ]

    R2s = np.array([Rsquare(inverse_gains, model(isos, *params)) for model, params in zip(models, parameters)])
    best = R2s.argmax()

    return models[best], model_errs[best], model_label[best], parameters[best], covariances[best], R2s[best]

def get_gain(table, iso):
    ind = np.where(table[0] == iso)[0]
    return table[1, ind][0], table[2, ind][0]
