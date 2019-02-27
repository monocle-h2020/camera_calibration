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


def _print_model_parameters(labels, parameters, errors):
    for label, parameter, error in zip(labels, parameters, errors):
        print(f"{label} = {parameter:.5f} +- {error:.5f}")


def fit_iso_normalisation_relation(isos, ratios, ratios_errs=None, min_iso=50, max_iso=50000):
    parameters_linear, covariance_linear = np.polyfit(isos, ratios, 1, cov=True)
    errors_linear = np.sqrt(np.diag(covariance_linear))
    model_linear = generate_linear_model(*parameters_linear)
    R2_linear = Rsquare(ratios, model_linear(isos))

    parameters_knee, covariance_knee = curve_fit(knee_model, isos, ratios, p0=[1/min_iso, 0, 200], bounds=([0, -np.inf, 1.05*min_iso], [1, np.inf, 0.95*max_iso]))
    errors_knee = np.sqrt(np.diag(covariance_knee))
    model_knee = generate_knee_model(*parameters_knee)
    R2_knee   = Rsquare(ratios, model_knee(isos))

    if R2_linear < 0.9 and R2_knee < 0.9:
        raise ValueError("Could not find an accurate (R^2 >= 0.9) fit to the iso-normalization relation")
    elif R2_linear >= 0.9:
        model, R2, parameters, errors, labels = model_linear, R2_linear, parameters_linear, errors_linear, ["Slope", "Offset"]
        print("Found linear model [y = ax + b]")
    elif R2_knee >= 0.9:
        model, R2, parameters, errors, labels = model_knee, R2_knee, parameters_knee, errors_knee, ["Slope", "Offset", "ISO cap"]
        print(f"Found knee model [y = ax + b, capped at K]")
    # space for extra models here
    else:
        raise ValueError("This should never occur -- are all comparisons the right way around?")

    _print_model_parameters(labels, parameters, errors)
    print(f"(R^2 = {R2:.6f})")

    return model, R2, parameters, errors


def normalise(data, iso, lookup_table):
    normalisation_factor = lookup_table[1][iso]
    new_data = data / normalisation_factor
    return new_data
