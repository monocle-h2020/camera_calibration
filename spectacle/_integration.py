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
