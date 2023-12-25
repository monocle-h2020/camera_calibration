"""
Code relating to dark current correction, such as fitting a trend or loading a map.
"""
from typing import Iterable, Optional

import numpy as np
from tqdm import tqdm

from . import io
from .general import return_with_filename


def fit_dark_current_linear(exposure_times: Iterable[float], data: np.ndarray[float], batch_size: int=500000, progressbar=True, leave_progressbar=False) -> tuple[np.ndarray[np.float64], np.ndarray[np.float64]]:
    """
    Fit a linear trend to each pixel in the data set `data` taken at varying `exposure_times`.
    To account for memory issues, fits are performed in batches, controlled by the batch_size (note that batches may differ in size slightly due to rounding).

    Both a bias (offset) and dark current (slope) are obtained from the linear fit.
    The bias is treated as a free parameter to account for variations and noise.
    """
    # The data are reshaped to a flat list for each exposure time, fitted, and then reshaped back to the original shape.
    original_shape = data.shape[1:]
    data_reshaped = data.reshape((data.shape[0], -1))

    # Divide the data into batches
    n_batches = data_reshaped.shape[1] // batch_size + 1
    data_split = np.array_split(data_reshaped, n_batches, axis=1)
    batch_size_real = data_reshaped.shape[1] / n_batches

    # Perform the polynomial fitting for each batch, then merge the results
    dark_fit, bias_fit = zip(*tqdm((np.polynomial.polynomial.polyfit(exposure_times, data_batch, 1) for data_batch in data_split), total=n_batches, desc="Fitting dark current", unit_scale=batch_size_real, unit="pixel", disable=not progressbar, leave=leave_progressbar))
    dark_fit, bias_fit = np.concatenate(dark_fit), np.concatenate(bias_fit)

    dark_reshaped = dark_fit.reshape(original_shape)
    bias_reshaped = bias_fit.reshape(original_shape)

    return dark_reshaped, bias_reshaped


def load_dark_current_map(root: io.Path | str, return_filename=False) -> tuple[np.ndarray[np.float64], Optional[io.Path]]:
    """
    Load the normalised dark current map located at root/`calibration/dark_current_normalised.npy`
    If `return_filename` is True, also return the exact filename used.
    """
    root = io.Path(root)
    filename = io.find_matching_file(root/"calibration", "dark_current_normalised.npy")
    dark_current_map = np.load(filename)
    return return_with_filename(dark_current_map, filename, return_filename)


def correct_dark_current_from_map(dark_current_map: np.ndarray[float], exposure_time: float | np.ndarray[float], data: np.ndarray[float]) -> np.ndarray[np.float64]:
    """
    Apply a dark current correction from a dark current map `dark_current_map`, multiplied by an `exposure_time`, to any number of elements in `data`.

    `exposure_time` can be an iterable (list or array) of exposure times, in which case it must be the same length as `data`.
    """
    # Check if `exposure_time` is iterable
    try:
        _ = iter(exposure_time)
    # If `exposure_time` was not iterable, assume it is a constant value
    except TypeError:
        dark_current = exposure_time * dark_current_map
    # If `exposure time` was an iterable, check that it has the same number of
    # elements as `data` and calculate the effective dark current for each
    else:
        assert len(exposure_time) == len(data), f"Exposure time is an iterable but has a different length ({len(exposure_time)}) than the data ({len(data)})."
        dark_current = exposure_time[:, np.newaxis, np.newaxis] * dark_current_map
    # In any case, correct the data
    finally:
        data_corrected = data - dark_current

    return data_corrected
