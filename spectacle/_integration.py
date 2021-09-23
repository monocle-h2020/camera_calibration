"""
This submodule contains functions related to numerical integration.
It is a helper module, to be imported elsewhere.
"""
import numpy as np
from scipy.linalg import block_diag
from ._interpolation import apply_interpolation_matrix as apply_integration_matrix

def trapezoid_matrix(x):
    """
    Generate the transformation matrix for the trapezoid rule.
    This is a column vector that looks like:
    0.5*[x1-x0, x2-x0, x3-x1, x4-x2, ..., xN-x(N-2), xN-x(N-1)]
    """
    # Generate the middle bit, which is just x(i) - x(i-2)
    middle = 0.5*(x[2:] - x[:-2])

    # Generate the start and end, which are separate
    start = [0.5*(x[1] - x[0])]
    end = [0.5*(x[-1] - x[-2])]

    # Combine them into one matrix and return the result
    combined = np.concatenate([start, middle, end])

    return combined


def simpson_matrix(x):
    """
    Generate the transformation matrix using Simpson's rule.
    This is a column vector.
    This method assumes regular spacing in x.
    """
    # Check if x is regularly spaced
    assert len(np.unique(np.diff(x))) == 1, "Input is not regularly spaced."

    # Normalisation factor
    h = np.diff(x)[0]

    # Generate a vector of all ones, then populate it with 2s and 4s, except the edges
    M = np.ones_like(x, dtype=np.float64)
    M[1:-1:2] = 4.
    M[2:-1:2] = 2.

    # Normalisation
    M *= (h/3)

    return M
