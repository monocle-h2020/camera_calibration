"""
This submodule contains functions related to spectral response functions and
colour analysis.
"""

import numpy as np
from matplotlib import pyplot as plt
from scipy.linalg import block_diag

from . import io, plot
from .general import return_with_filename
from ._xyz import wavelengths as cie_wavelengths, xyz as cie_xyz
from ._spectral_convolution import convolve, convolve_multi
from ._interpolation import interpolation_functions, apply_interpolation_matrix, linear_interpolation, linear_interpolation_matrix
from ._integration import trapezoid_matrix, simpson_matrix
from ._monochromator import load_cal_NERC, load_monochromator_data, load_monochromator_data_multiple, generate_slices_for_RGBG2_bands, flatten_monochromator_image_data, flatten_monochromator_image_data_multiple

wavelengths_interpolated = np.arange(390, 701, 1)

# Matrix for converting RGBG2 to RGB
M_RGBG2_to_RGB = np.array([[1, 0  , 0, 0  ],
                           [0, 0.5, 0, 0.5],
                           [0, 0  , 1, 0  ]])


def effective_bandwidth(wavelengths, response, axis=0, **kwargs):
    """
    Calculate the effective bandwidth of a spectral band.
    """
    response_normalised = response / response.max(axis=axis)
    return np.trapz(response_normalised, x=wavelengths, axis=axis, **kwargs)


def interpolate(wavelengths, response, interpolate_to=wavelengths_interpolated):
    interpolated = np.stack([np.interp(interpolate_to, wavelengths, R) for R in response.T]).T
    return interpolate_to, interpolated


def repeat_matrix(M, nr_bands=4):
    """
    Repeat a matrix nr_bands times along the diagonal.
    For example, repeat an interpolation matrix generated for a single band
    4 times, to extend it to RGBG2.
    """
    M_repeated = block_diag(*[M]*nr_bands)

    return M_repeated


def plot_monochromator_curves(wavelengths, mean, variance, wavelength_min=390, wavelength_max=700, unit="ADU", title="", saveto=None):
    """
    Plot spectral response curves as measured by monochromator.
    Plots three panels, namely mean response, variance, and signal-to-noise ratio.
    """
    # Labels for the plots
    labels = [f"Response\n[{unit}]", f"Variance\n[{unit}$^2$]", "SNR"]

    # Make the figure
    fig, axs = plt.subplots(nrows=3, sharex=True, figsize=(4,4))

    # Plot the mean and variance
    plot._rgbgplot(wavelengths, mean, axs[0].plot)
    plot._rgbgplot(wavelengths, variance, axs[1].plot)

    # Calculate and plot the signal-to-noise ratio
    SNR = mean/np.sqrt(variance)
    plot._rgbgplot(wavelengths, SNR, axs[2].plot)

    # Plot parameters
    axs[0].set_xlim(wavelength_min, wavelength_max)
    axs[0].set_ylim(ymin=0)
    axs[0].set_title(title)
    axs[-1].set_xlabel("Wavelength (nm)")

    for ax, label in zip(axs, labels):
        ax.grid(ls="--")
        ax.set_ylabel(label)

    # Save/show result
    plot._saveshow(saveto)


def load_spectral_response(root, return_filename=False):
    """
    Load the spectral response curves located at
    `root`/calibration/spectral_response.csv.

    If `return_filename` is True, also return the exact filename used.
    """
    # Try to use a CSV file
    filename = io.find_matching_file(root/"calibration", "spectral_response.csv")
    spectral_response = np.loadtxt(filename, delimiter=",").T

    return return_with_filename(spectral_response, filename, return_filename)


def plot_spectral_responses(wavelengths, SRFs, labels, linestyles=["-", "--", ":", "-."], xlim=(390, 700), ylim=(0,1.02), ylabel="Relative sensitivity", saveto=None, **kwargs):
    """
    Plot spectral responses (`SRFs`) together in one panel.
    """
    # Create a figure to hold the plot
    plt.figure(figsize=(7,3), tight_layout=True)

    # Loop over the response curves
    for wavelength, SRF, label, style in zip(wavelengths, SRFs, labels, linestyles):
        plot._rgbgplot(wavelength, SRF, ls=style, **kwargs)

        # Add an invisible line for the legend
        plt.plot([-1000,-1001], [-1000,-1001], c='k', ls=style, label=label, **kwargs)

    # Plot parameters
    plt.grid(ls="--")
    plt.xticks(np.arange(0,1000,50))
    plt.xlim(*xlim)
    plt.xlabel("Wavelength [nm]")
    plt.ylabel(ylabel)
    plt.ylim(*ylim)
    plt.legend(loc="best")
    plot._saveshow(saveto)


def load_spectral_bands(root, return_filename=False):
    """
    Load the effective spectral bandwidths located at
    `root`/calibration/spectral_bands.csv.

    If `return_filename` is True, also return the exact filename used.
    """
    filename = io.find_matching_file(root/"calibration", "spectral_bands.csv")
    spectral_bands = np.loadtxt(filename, delimiter=", ").T

    return return_with_filename(spectral_bands, filename, return_filename)


def interpolate_spectral_data(old_wavelengths, old_data, new_wavelengths, **kwargs):
    """
    Interpolate spectral data `old_data` at `old_wavelengths` to a set of
    `new_wavelengths`. Handles multi-channel (RGB or RGBG2) data.

    Assumes the `old_data` have the shape (number_of_channels, number_of_wavelengths)

    Any additional **kwargs are passed to numpy.interp
    """
    # Interpolate the data separately in a list comprehension
    interpolated_data = [np.interp(new_wavelengths, old_wavelengths, channel, **kwargs) for channel in old_data]

    # Stack the interpolated data into a numpy array
    interpolated_data = np.stack(interpolated_data)

    return interpolated_data


def array_slice(a, axis, start, end, step=1):
    """
    Slice an array along an arbitrary axis.
    Adapted from https://stackoverflow.com/a/64436208/2229219
    """
    return (slice(None),) * (axis % a.ndim) + (slice(start, end, step),)


def convert_between_colourspaces(data, conversion_matrix, axis=0):
    """
    Convert data from one colour space to another, using the given conversion matrix.
    Matrix multiplication can be done on an arbitrary axis in the data, using np.tensordot.
    Example uses include converting from RGBG2 to RGB and from RGB to XYZ.
    """
    # Use tensor multiplication to multiply along an arbitrary axis
    new_data = np.tensordot(conversion_matrix, data, axes=((1), (axis)))
    new_data = np.moveaxis(new_data, 0, axis)

    return new_data


def convert_RGBG2_to_RGB(RGBG2_data, axis=0):
    """
    Convert data in Bayer RGBG2 format to RGB format, by averaging the G and G2
    channels.

    The RGBG2 data are assumed to have their colour axis on `axis`. For example,
    if `axis=0`, then the data are assumed to have a shape of (4, ...).
    If `axis=2`, then the data are assumed to be (:, :, 4, ...). Et cetera.

    The resulting array has the same shape but with the given axis changed from
    4 to 3.
    """
    RGB_data = convert_between_colourspaces(RGBG2_data, M_RGBG2_to_RGB, axis=axis)

    return RGB_data


def convert_RGBG2_to_RGB_uncertainties(uncertainties_RGBG2):
    """
    Convert uncertainties from RGBG2 to RGB.

    Assumes the `uncertainties_RGBG2` have the shape (4, number_of_wavelengths)
    """
    # Make a new array containing only the RGB data
    uncertainties_RGB = uncertainties_RGBG2.copy()[:3]

    # Replace the G axis with the average of G and G2
    uncertainties_RGB[1] = 0.5 * np.sqrt(uncertainties_RGBG2[1]**2 + uncertainties_RGBG2[3]**2)

    return uncertainties_RGB


def correct_spectra(spectral_response_wavelengths, spectral_response, data_wavelengths, data, axis_RGBG2=0, axis_wavelengths=-1):
    """
    Correct any number of spectra `data` for the `spectral response` interpolated to
    the data wavelengths. Note that the arrays in data must share the same wavelengths.

    The spectral responses are interpolated to the wavelengths given by the
    user. Spectral responses outside the range of the calibration data are
    assumed to be 0.

    The correction can be done on any axes given by the keywords `axis_RGBG2` and `axis_wavelengths`.
    The default `axis_RGBG2` is 0, corresponding to an array of shape (3, ...) or (4, ...).
    The default `axis_wavelengths` is 1, corresponding to an array of shape (:, L, ...) with L the length of `data_wavelengths`.
    """
    # Check that the data are the right shape
    assert data.shape[axis_wavelengths] == data_wavelengths.shape[0], f"Wavelengths ({data_wavelengths.shape[0]}) and data ({data.shape[axis_wavelengths]}) have different numbers of wavelength values."
    assert data.shape[axis_RGBG2] in (3, 4), f"Incorrect number of channels ({data.shape[axis_RGBG2]}) in data; expected 3 (RGB) or 4 (RGBG2)."

    # Convert the spectral response to the same shape as the input data
    spectral_response_interpolated = interpolate_spectral_data(spectral_response_wavelengths, spectral_response, data_wavelengths, left=0, right=0)

    # Convert the spectral response into the correct channels (RGB or RGBG2)
    if data.shape[axis_RGBG2] == 3:  # RGB data
        spectral_response_final = convert_RGBG2_to_RGB(spectral_response_interpolated, axis=0)
    else:  # RGBG2 data
        spectral_response_final = spectral_response_interpolated

    # Correct the spectra
    # Numpy's broadcasting works when the matching axes are at the end.
    sources = (axis_RGBG2, axis_wavelengths)
    destinations = (-2, -1)
    data_normalised = np.moveaxis(data.copy(), sources, destinations)
    data_normalised /= spectral_response_final
    data_normalised = np.moveaxis(data_normalised, destinations, sources)

    return data_normalised


def effective_wavelengths(wavelengths, spectral_responses):
    """
    Calculate the effective wavelength of each band in `spectral_responses` by
    taking a weighted mean over the spectral range.
    """
    # Calculate the weighted mean
    weighted_means = [np.average(wavelengths, weights=spectral_band) for spectral_band in spectral_responses]

    return weighted_means


def calculate_XYZ_matrix(wavelengths, spectral_response):
    """
    Calculate the matrix used to convert data from a camera with given
    `spectral_response` curves to CIE XYZ colour space.

    `spectral_response` can have 3 (RGB) or 4 (RGBG2) channels. In the RGBG2
    case, it is first converted to RGB.

    The data are assumed to consist of 3 (RGB) or 4 (RGBG2) rows and a column
    for every wavelength. If not, an error is thrown.
    """
    # Convert the input spectral response to RGB
    assert spectral_response.shape[0] in (3, 4), f"Incorrect number of channels ({spectral_response.shape[0]}) in data; expected 3 (RGB) or 4 (RGBG2)."
    if spectral_response.shape[0] == 3:  # Already RGB
        spectral_response_RGB = spectral_response.copy()
    else:  # RGBG2
        spectral_response_RGB = convert_RGBG2_to_RGB(spectral_response)

    # Interpolate the spectral response to the CIE XYZ wavelengths
    SRF_RGB_interpolated = interpolate_spectral_data(wavelengths, spectral_response_RGB, cie_wavelengths)

    # Convolve the SRFs and XYZ curves
    # Resulting matrix:
    # [X_R  X_G  X_B]
    # [Y_R  Y_G  Y_B]
    # [Z_R  Z_G  Z_B]
    SRF_XYZ_product = np.einsum("xw,rw->xr", cie_xyz, SRF_RGB_interpolated) / len(cie_wavelengths)

    # Normalise by column
    SRF_xyz = SRF_XYZ_product / SRF_XYZ_product.sum(axis=0)
    white_E = np.array([1., 1., 1.])  # Equal-energy illuminant E
    normalisation_vector = np.linalg.inv(SRF_xyz) @ white_E
    normalisation_matrix = np.identity(3) * normalisation_vector
    M_RGB_to_XYZ = SRF_xyz @ normalisation_matrix

    return M_RGB_to_XYZ


def _find_matching_axis(data, axis_length):
    """
    Find an axis in `data` that has the given length `axis_length`.
    """
    matching_axes = [i for i, length in enumerate(data.shape) if length == axis_length]

    if len(matching_axes) == 0:
        raise ValueError(f"No axis in the given data array matches the given axis length `{axis_length}`. Array shape: {data.shape}")
    elif len(matching_axes) >= 2:
        raise ValueError(f"No axis in the given data array matches the given axis length `{axis_length}`. Array shape: {data.shape}")
    else:
        return matching_axes[0]


def convert_to_XYZ(RGB_to_XYZ_matrix, RGB_data, axis=None):
    """
    Convert RGB data to XYZ. The RGB data can be multi-dimensional, for example a
    spectrum (3, L) with L the number of wavelengths or an image (3, X, Y) with
    X and Y the number of pixels in either direction.
    If the axis is not specified by the user, an axis with a length of 3 is searched for.
    If 0 or >=2 such axes are found, an error is raised.

    RGBG2 data can also be passed, in which case an axis with length 4 must be given or
    will be looked for. The data will then first be converted to RGB which adds computation time.
    """
    # Convert the data to a numpy array
    RGB_data = np.array(RGB_data)

    if axis is None:  # If no axis is supplied, look for one
        try:  # First see if the data are RGB
            axis = _find_matching_axis(RGB_data, 3)
        except ValueError:  # If not, try RGBG2
            axis = _find_matching_axis(RGB_data, 4)
    else:  # If an axis was supplied, check that is has the correct length
        assert RGB_data.shape[axis] in (3,4), f"The given axis ({axis}) in the data array has a length ({RGB_data.shape[axis]}) that is not 3 or 4."

    # If RGBG2 data were given, first convert the data to RGB
    if RGB_data.shape[axis] == 4:
        RGB_data = convert_RGBG2_to_RGB(RGB_data, axis=axis)

    # Perform the matrix multiplication
    XYZ_data = convert_between_colourspaces(RGB_data, RGB_to_XYZ_matrix, axis=axis)

    return XYZ_data


def load_XYZ_matrix(root, return_filename=False):
    """
    Load an RGB -> XYZ conversion matrix located at
    `root`/calibration/RGB_to_XYZ_matrix.csv.

    If `return_filename` is True, also return the exact filename used.
    """
    filename = io.find_matching_file(root/"calibration", "RGB_to_XYZ_matrix.csv")
    XYZ_matrix = np.loadtxt(filename, delimiter=", ")

    return return_with_filename(XYZ_matrix, filename, return_filename)


def calculate_xy_base_vectors(XYZ_matrix):
    """
    Calculate the base vectors in xy chromaticity space for a given conversion matrix M.
    """
    xyz_matrix = XYZ_matrix / XYZ_matrix.sum(axis=0)  # Divide by X+Y+Z per column
    base_vectors = np.hsplit(xyz_matrix, 3)
    base_xy = [vector[:2].T[0] for vector in base_vectors]

    return base_xy


def plot_xy_on_gamut(xy_base_vectors, label="", saveto=None):
    """
    Plot the xy base vectors of any number of colour spaces on the xy plane.
    If possible, use colorio's function to plot the human eye and sRGB gamuts.
    """
    saveto = plot._convert_to_path(saveto)

    try:
        from colorio._tools import plot_xy_gamut
    except ImportError:
        print("Could not import colorio, using simple gamut plot.")
        plt.xlim(0, 0.8)
        plt.ylim(0, 0.8)
        plt.xlabel("x")
        plt.ylabel("y")
        sRGB_triangle = plt.Polygon([[0.64,0.33], [0.30, 0.60], [0.15, 0.06]], fill=True, linestyle="-", label="sRGB")
        plt.gca().add_patch(sRGB_triangle)
    else:
        plot_xy_gamut()

    # Check if a single set of base vectors was given or multiple
    if len(xy_base_vectors[0]) != 3:  # A single set of base vectors
        xy_base_vectors = [xy_base_vectors]
        label = [label]
        plt.title(f"{label[0]} colour space\ncompared to sRGB")
    else:  # If multiple sets were given
        nr_sets = len(xy_base_vectors)
        # If no or insufficient labels were provided, warn the user, and provide empty strings instead
        if len(label) != nr_sets:
            print(f"{len(label)} labels were provided for {nr_sets} data sets. Using empty labels instead.")
            label = [""] * nr_sets
        plt.title(f"Colour spaces\ncompared to sRGB")

    # kwargs to make triangles look distinct
    kwargs = [{"linestyle": "dashed"},
              {"linestyle": "dotted"},
              {"linestyle": "dashdot"},
              {"linestyle": "solid", "linewidth": 0.5}]

    triangles = [plt.Polygon(base_vectors, fill=False, label=label_single, **kwargs_single) for base_vectors, label_single, kwargs_single in zip(xy_base_vectors, label, kwargs)]
    for triangle in triangles:
        plt.gca().add_patch(triangle)

    plt.legend(loc="upper right")

    plot._saveshow(saveto)


def plot_xyz_and_rgb_single(ax, wavelengths, responses, label="", legend_labels="rgb"):
    """
    Plot the given spectral responses as a function of wavelength into the given axes object `ax`.
    Put the `label` on the y axis.
    """
    kwargs = {"lw": 3}
    colours = ["#d95f02", "#1b9e77", "#7570b3"]
    for response, letter, colour in zip(responses, legend_labels, colours):
        ax.plot(wavelengths, response, c=colour, label=letter, **kwargs)
    ax.set_ylabel(f"{label}\nresponse")
    ax.legend(loc="upper left", bbox_to_anchor=(1,1))
    ax.set_xlim(390, 700)
    ax.set_ylim(ymin=0)


def plot_xyz_and_rgb(RGB_wavelengths, RGB_responses, label="", saveto=None):
    """
    Plot the xyz colour matching functions and given RGB responses.
    """
    # Check if a single set of wavelengths/responses was given or multiple
    try:  # This throws an error if a single set of wavelengths was given
        _ = len(RGB_wavelengths[0])
    except TypeError:  # If a single set of wavelengths was given, make them into a list
        RGB_wavelengths = [RGB_wavelengths]
        RGB_responses = [RGB_responses]
        label = [label]
        nr_sets = 1
    else:  # If multiple sets were given
        assert len(RGB_wavelengths) == len(RGB_responses), f"Different numbers of wavelength sets ({len(RGB_wavelengths)}) and response sets ({len(RGB_responses)}) were provided."
        nr_sets = len(RGB_wavelengths)
        # If no or insufficient labels were provided, warn the user, and provide empty strings instead
        if len(label) != nr_sets:
            print(f"{len(label)} labels were provided for {nr_sets} data sets. Using empty labels instead.")
            label = [""] * nr_sets

    fig, axs = plt.subplots(nrows=1+nr_sets, figsize=(4,1.5*(nr_sets+1)), sharex=True)
    plot_xyz_and_rgb_single(axs[0], cie_wavelengths, cie_xyz, label="CIE XYZ", legend_labels=["$\\bar x$", "$\\bar y$", "$\\bar z$"])
    for ax, wavelengths, responses, label_single in zip(axs[1:], RGB_wavelengths, RGB_responses, label):
        plot_xyz_and_rgb_single(ax, wavelengths, responses, label=label_single)
    axs[-1].set_xlabel("Wavelength [nm]")

    plot._saveshow(saveto)
