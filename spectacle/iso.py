"""
Code relating to ISO speed normalisation, such as generating or reading
look-up tables.
"""

import numpy as np
from scipy.optimize import curve_fit
from .general import Rsquare


def generate_linear_model(slope, offset):
    """
    Generate a linear ISO speed model, with two parameters: slope and offset.
    A function for applying this model is returned.
    """
    model = lambda isos: np.polyval([slope, offset], isos)
    return model


def knee_model(isos, slope, offset, knee):
    """
    Apply a 'knee' ISO speed model, with three parameters: slope, offset,
    and knee. The model is linear up to the `knee` ISO speed, then constant.
    """
    linear_model = generate_linear_model(slope, offset)
    values = np.clip(linear_model(isos), a_min=None, a_max=linear_model(knee))
    return values


def generate_knee_model(slope, offset, knee):
    """
    Generate a 'knee' ISO speed model, with three parameters: slope, offset,
    and knee. The model is linear up to the `knee` ISO speed, then constant.
    A function for applying this model is returned.
    """
    model = lambda isos: knee_model(isos, slope, offset, knee)
    return model


def _print_model_parameters(labels, parameters, errors):
    """
    Print the parameters of a normalisation model.
    """
    for label, parameter, error in zip(labels, parameters, errors):
        print(f"{label} = {parameter:.5f} +- {error:.5f}")


# Dict mapping names to functions for generating ISO normalisation models
model_generator = {"Linear": generate_linear_model, "Knee": generate_knee_model}


def fit_iso_normalisation_relation(isos, ratios, ratios_errs=None, min_iso=50, max_iso=50000):
    """
    Fit a relation between ISO speed and normalisation. Currently two types of
    model are supported, namely linear and 'knee'-type. Both are fitted and
    the best is chosen.
    """
    # Fit a linear model
    parameters_linear, covariance_linear = np.polyfit(isos, ratios, 1, cov=True)
    errors_linear = np.sqrt(np.diag(covariance_linear))
    model_linear = generate_linear_model(*parameters_linear)
    R2_linear = Rsquare(ratios, model_linear(isos))

    # Fit a knee-type model
    parameters_knee, covariance_knee = curve_fit(knee_model, isos, ratios, p0=[1/min_iso, 0, 200], bounds=([0, -np.inf, 1.05*min_iso], [1, np.inf, 0.95*max_iso]))
    errors_knee = np.sqrt(np.diag(covariance_knee))
    model_knee = generate_knee_model(*parameters_knee)
    R2_knee   = Rsquare(ratios, model_knee(isos))

    # Use R2 (oh no) to choose the best model. Linear is preferred if it is
    # adequate, otherwise knee-type is used.
    if R2_linear < 0.9 and R2_knee < 0.9:
        raise ValueError("Could not find an accurate (R^2 >= 0.9) fit to the iso-normalization relation")
    elif R2_linear >= 0.9:
        model, R2, parameters, errors, labels, model_type = model_linear, R2_linear, parameters_linear, errors_linear, ["Slope", "Offset"], "Linear"
        print("Found linear model [y = ax + b]")
    elif R2_knee >= 0.9:
        model, R2, parameters, errors, labels, model_type = model_knee, R2_knee, parameters_knee, errors_knee, ["Slope", "Offset", "ISO cap"], "Knee"
        print(f"Found knee model [y = ax + b, capped at K]")
    # space for extra models here
    else:
        raise ValueError("This should never occur -- are all comparisons the right way around?")

    # Print and return the model parameters
    _print_model_parameters(labels, parameters, errors)
    print(f"(R^2 = {R2:.6f})")

    return model_type, model, R2, parameters, errors


def normalise_single_iso(data, iso, lookup_table):
    """
    Normalise data at a single ISO speed using the look-up table.
    """
    normalisation_factor = lookup_table[1][iso]
    new_data = data / normalisation_factor
    return new_data


def normalise_multiple_iso(data, isos, lookup_table):
    """
    Normalise data at multiple ISO speeds using the look-up table.
    `data` and `isos` are assumed to have the same length, i.e. each element
    of `data` has one associated ISO speed in `isos`.
    """
    as_list = [normalise_single_iso(data_sub, ISO, lookup_table) for data_sub, ISO in zip(data, isos)]
    as_array = np.array(as_list)
    return as_array


def normalise_iso_general(lookup_table, isos, data):
    """
    Normalise data for ISO speed in general. Uses either `normalise_single_iso`
    or `normalise_multiple_iso` based on the number of isos given.
    """
    if isinstance(isos, (int, float)):
        data_normalised = normalise_single_iso  (data, isos, lookup_table)
    else:
        data_normalised = normalise_multiple_iso(data, isos, lookup_table)

    return data_normalised


def load_iso_lookup_table(root, return_filename=False):
    """
    Load the ISO normalization lookup table located at
    `root`/calibration/iso_normalisation_lookup_table.csv

    If `return_filename` is True, also return the exact filename the table
    was retrieved from.
    """
    filename = root/"calibration/iso_normalisation_lookup_table.csv"
    table = np.loadtxt(filename, delimiter=",").T
    if return_filename:
        return table, filename
    else:
        return table


def load_iso_model(root, return_filename=False):
    """
    Load the ISO normalization function, the parameters of which are contained
    in `root`/calibration/iso_normalisation_model.csv

    If `return_filename` is True, also return the exact filename the model
    was retrieved from.

    To do: include in ISO model object
    """
    filename = root/"calibration/iso_normalisation_model.csv"
    as_array = np.loadtxt(filename, dtype=str, delimiter=",")

    model_type = as_array[0]
    if model_type == "Linear":
        parameters = as_array[1:3]
        errors = as_array[3:]
    elif model_type == "Knee":
        parameters = as_array[1:4]
        errors = as_array[4:]
    else:
        raise ValueError(f"Unknown model type `{model_type}` in file `{filename}`.")

    parameters = parameters.astype(np.float64)
    errors     = errors.astype(np.float64)
    model = model_generator[model_type](*parameters)

    if return_filename:
        return model, filename
    else:
        return model


def save_iso_model(saveto, model_type, parameters, errors):
    """
    Save the parameters to the ISO normalisation function to `saveto`.

    To do: include in ISO model object
    """

    model_array = np.array([model_type, *parameters, *errors])
    model_array = model_array[:, np.newaxis].T

    if model_type == "Linear":
        header = "Model, a, b, a_err, b_err"
    elif model_type == "Knee":
        header = "Model, a, b, K, a_err, b_err, K_err"
    else:
        raise ValueError(f"Unknown model type `{model_type}`.")

    np.savetxt(saveto, model_array, fmt="%s", delimiter=",", header=header)


def load_iso_data(root, return_filename=False):
    """
    Load ISO normalisation data from
    `root`/intermediaries/iso_normalisation/iso_data.npy

    If `return_filename` is True, also return the exact filename the data
    were retrieved from.
    """
    filename = root/"intermediaries/iso_normalisation/iso_data.npy"
    data = np.load(filename)
    if return_filename:
        return data, filename
    else:
        return data
