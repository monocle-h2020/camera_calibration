import numpy as np
from scipy.optimize import curve_fit
from phonecal.general import Rsquare


def generate_linear_model(slope, offset):
    model = lambda isos: np.polyval([slope, offset], isos)
    return model


def knee_model(isos, slope, offset, knee):
    linear_model = generate_linear_model(slope, offset)
    values = np.clip(linear_model(isos), a_min=None, a_max=linear_model(knee))
    return values


def generate_knee_model(slope, offset, knee):
    model = lambda isos: knee_model(isos, slope, offset, knee)
    return model


def fit_iso_normalisation_relation(isos, ratios, ratios_errs=None, min_iso=50, max_iso=50000):
    parameters_linear, covariance_linear = np.polyfit(isos, ratios, 1, cov=True)
    model_linear = generate_linear_model(*parameters_linear)
    R2_linear = Rsquare(ratios, model_linear(isos))

    parameters_knee, covariance_knee = curve_fit(knee_model, isos, ratios, p0=[1/min_iso, 0, 200], bounds=([0, -np.inf, 1.05*min_iso], [1, np.inf, 0.95*max_iso]))
    model_knee = generate_knee_model(*parameters_knee)
    R2_knee   = Rsquare(ratios, model_knee(isos))

    if R2_linear < 0.9 and R2_knee < 0.9:
        raise ValueError("Could not find an accurate (R^2 >= 0.9) fit to the iso-normalization relation")
    elif R2_linear >= 0.9:
        model = model_linear
        R2 = R2_linear
        print(f"Found linear model [y = ax + b] with a = {parameters_linear[0]:.3f} & b = {parameters_linear[1]:.3f}", end=" ")
    elif R2_knee >= 0.9:
        model = model_knee
        R2 = R2_knee
        print(f"Found knee model with with a = {parameters_knee[0]:.3f} & b = {parameters_knee[1]:.3f} & k = {parameters_knee[2]:.1f}", end=" ")
    # space for extra models here
    else:
        raise ValueError("This should never occur -- are all comparisons the right way around?")

    print(f"(R^2 = {R2:.6f})")

    return model, R2


def normalise(data, iso, lookup_table):
    normalisation_factor = lookup_table[1][iso]
    new_data = data / normalisation_factor
    return new_data
