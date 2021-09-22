"""
This submodule contains functions related to interpolation.
It is a helper module, to be imported elsewhere.
"""
import numpy as np
from scipy.linalg import block_diag


def apply_interpolation_matrix(M, y, covariance=None):
    """
    Apply a given interpolation matrix M to a data set y and optionally to
    a covariance matrix.
    """
    y_new = M @ y
    if covariance is not None:
        covariance_new = M @ covariance @ M.T
        return y_new, covariance_new
    else:
        return y_new


def linear_interpolation_matrix(x_new, x, debug=False):
    """
    Calculate the transformation matrix for a linear interpolation.
    """
    # Start with an array of all zeros
    M = np.zeros((x_new.shape[0], x.shape[0]))
    indices_new = np.arange(x_new.shape[0])

    # For each element in x_new, find the corresponding elements in x that are
    # directly to its left and right. These are the ones to interpolate between.
    i1 = np.searchsorted(x, x_new, side="left")  # index to the left of our interpolated x
    i1 = np.clip(i1, 0, len(x)-1)  # Ensure nothing goes wrong at the edges
    i0 = i1 - 1

    # This prints output which can be used to check that the index selection works.
    if debug:
        for i, w_new in enumerate(x_new):
            print(f"New index {i:>3} \t Old index {i0[i]:>3} \t {x[i0[i]]:.1f} -- {w_new:.1f} -- {x[i1[i]]:.1f}")

    # Calculate the weighting terms corresponding to i0 and i1
    x0 = x[i0]
    x1 = x[i1]
    fraction = (x_new - x0)/(x1 - x0)
    y0_terms = 1 - fraction
    y1_terms = fraction

    # Insert the weighting terms into the matrix
    M[indices_new,i0] = y0_terms
    M[indices_new,i1] = y1_terms

    return M


def linear_interpolation(x_new, x, y, covariance=None, debug=False):
    """
    Perform a linear interpolation using the matrix method.
    """
    # Calculate the transformation matrix
    M = linear_interpolation_matrix(x_new, x, y, debug=debug)

    # Apply M and return the result
    return apply_interpolation_matrix(M, y, covariance=covariance)


interpolation_functions = {"linear": linear_interpolation}
