import numpy as np
from . import calibrate, io, raw

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
        mean_RGBG, _ = raw.pull_apart(m, camera.bayer_map)

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

