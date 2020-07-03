import numpy as np
from matplotlib import pyplot as plt
from . import calibrate, io, raw, plot
from .xyz import wavelengths as cie_wavelengths, x as cie_x, y as cie_y, z as cie_z
from warnings import warn

def effective_bandwidth(wavelengths, response, axis=0, **kwargs):
    response_normalised = response / response.max(axis=axis)
    return np.trapz(response_normalised, x=wavelengths, axis=axis, **kwargs)


wavelengths_interpolated = np.arange(390, 701, 1)


def interpolate(wavelengths, response, interpolate_to=wavelengths_interpolated):
    interpolated = np.stack([np.interp(interpolate_to, wavelengths, R) for R in response.T]).T
    return interpolate_to, interpolated


def load_cal_NERC(filename, norm=True):
    """
    Function for loading NERC calibration files. A different function may be
    necessary for other calibration file formats.
    """
    data = np.genfromtxt(filename, skip_header=1, skip_footer=10)
    if norm:
        data = data / data.max()  # normalise to 1
    with open(filename, "r") as file:
        info = file.readlines()[0].split(",")
    start, stop, step = [float(i) for i in info[3:6]]
    wavelengths = np.arange(start, stop+step, step)
    arr = np.stack([wavelengths, data])
    return arr


def load_monochromator_data(root, folder, blocksize=100):
    """
    Load monochromator data, stored as a stack (mean/std) per wavelength in
    `folder`. For each wavelength, load the data, apply a bias correction, and
    take the mean and std of the central `blocksize`x`blocksize` pixels.
    Return the wavelengths with assorted mean values and standard deviations.
    """
    # Find the filenames
    mean_files = sorted(folder.glob("*_mean.npy"))
    stds_files = sorted(folder.glob("*_stds.npy"))
    assert len(mean_files) == len(stds_files)

    # Load metadata
    camera = io.load_metadata(root)
    bias = calibrate.load_bias_map(root)

    # Half-blocksize, to slice the arrays with
    d = blocksize//2

    # Empty arrays to hold the output
    wvls  = np.zeros((len(mean_files)))
    means = np.zeros((len(mean_files), 4))
    stds  = means.copy()

    # Loop over all files
    for j, (mean_file, stds_file) in enumerate(zip(mean_files, stds_files)):
        # Load the mean data
        m = np.load(mean_file)

        # Bias correction; don't use calibrate.correct_bias to prevent loading
        # the data from file every time
        m = m - bias

        # Demosaick the data
        mean_RGBG = camera.demosaick(m)

        # Select the central blocksize x blocksize pixels
        midx, midy = np.array(mean_RGBG.shape[1:])//2
        sub = mean_RGBG[:,midx-d:midx+d+1,midy-d:midy+d+1]

        # Take the mean value per Bayer channel
        m = sub.mean(axis=(1,2))

        # NaN if a channel's mean value is near saturation
        m[m >= 0.95 * camera.saturation] = np.nan

        # Store results
        means[j] = m
        stds[j] = sub.std(axis=(1,2))
        wvls[j] = mean_file.stem.split("_")[0]

        print(wvls[j], end=" ")

    print(folder)

    spectrum = np.stack([wvls, *means.T, *stds.T]).T
    return spectrum


def plot_monochromator_curves(wavelength, mean, std, wavelength_min=390, wavelength_max=700, unit="ADU", title="", saveto=None):
    plt.figure(figsize=(10,5))
    # Loop over the provided spectra
    for m, s in zip(mean, std):
        # Loop over the RGBG2 channels
        for j, c in enumerate("rybg"):
            # Plot the mean response per wavelength
            plt.plot(wavelength, m[:,j], c=c)

            # Plot the error per wavelength as a shaded area around the mean
            plt.fill_between(wavelength, m[:,j]-s[:,j], m[:,j]+s[:,j], color=c, alpha=0.3)

    # Plot parameters
    plt.xticks(np.arange(0, 1000, 50))
    plt.xlim(wavelength_min, wavelength_max)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel(f"Spectral response ({unit})")
    plt.ylim(ymin=0)
    plt.title(title)
    plt.grid(True)
    plot._saveshow(saveto)


def load_spectral_response(root, return_filename=False):
    """
    Load the spectral response curves located at
    `root`/calibration/spectral_response.csv.

    If no CSV is available, try an NPY file for backwards compatibility.
    This is deprecated and will no longer be supported in future releases.

    If `return_filename` is True, also return the exact filename the bias map
    was retrieved from.
    """
    # Try to use a CSV file
    filename = root/"calibration/spectral_response.csv"
    try:
        spectral_response = np.loadtxt(filename, delimiter=",").T

    # If no CSV file is available, check for an NPY file (deprecated)
    except IOError:
        try:
            filename = root/"calibration/spectral_response.npy"
            spectral_response = np.load(filename)

        # If still no luck - don't load anything, return an error
        except FileNotFoundError:
            raise IOError(f"Could not load CSV or NPY spectral response file from {root/'calibration/'}.")

        # If an NPY file was used instead of a CSV file, raise a warning about deprecation
        else:
            warn("NPY-format spectral response curves are deprecated and will no longer be supported in future releases.", DeprecationWarning)

    print(f"Using spectral response curves from '{filename}'")

    if return_filename:
        return spectral_response, filename
    else:
        return spectral_response


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


def convert_RGBG2_to_RGB(RGBG2_data):
    """
    Convert data in Bayer RGBG2 format to RGB format, by averaging the G and G2
    channels.

    Assumes the `RGBG2_data` have the shape (4, number_of_wavelengths)

    To do:
        - Error propagation
    """
    # Split the channels
    R, G, B, G2 = RGBG2_data

    # Take the average of the G and G2 channels
    G_combined = np.mean([G, G2], axis=0)

    # Stack the new RGB responses together and return the result
    RGB_data = np.stack([R, G_combined, B])

    return RGB_data


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
