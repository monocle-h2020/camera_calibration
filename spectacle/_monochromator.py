"""
This submodule contains functions related to monochromator data.
These functions are accessible from spectacle.spectral.
"""
import numpy as np

from . import io

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


def apply_calibration_NERC(calibration_data, data_wavelengths, data):
    """
    Divide the given data by the calibration data along matching wavelengths.
    """
    # Find the matching wavelengths and corresponding indices
    _, calibration_indices, _ = np.intersect1d(calibration_data[0], data_wavelengths, return_indices=True)

    # Get the corresponding calibration values
    calibration = calibration_data[1, calibration_indices]

    # Move the matching axis to the end so numpy can do broadcasting, divide, then move it back
    data_calibrated = np.moveaxis(data, 0, -1)
    data_calibrated = data_calibrated / calibration
    data_calibrated = np.moveaxis(data_calibrated, -1, 0)

    return data_calibrated


def apply_calibration_NERC_multiple(calibration_data, data_wavelengths, data):
    """
    Loop over the input arguments and apply apply_calibration_NERC to each.
    """
    # This could be made a lot shorter using splats but now it's easier to understand
    data_calibrated = [apply_calibration_NERC(cal, wvl, dat) for cal, wvl, dat in zip(calibration_data, data_wavelengths, data)]

    return data_calibrated



def load_monochromator_data(camera, folder, blocksize=100, flatfield=False):
    """
    Load monochromator data, stored as a stack (mean/std) per wavelength in
    `folder`. For each wavelength, load the data, apply a bias correction, and
    take the mean and std of the central `blocksize`x`blocksize` pixels.
    The `blocksize` is for the mosaicked image - when demosaicked, the RGBG2
    channels will be half its size each.

    Apply a bias correction and optionally a flat-field correction.

    Return the wavelengths with assorted mean values and standard deviations.
    """
    print(f"Loading monochromator data from `{folder}`...")

    # Central slice
    center = camera.central_slice(blocksize, blocksize)

    # Load all files
    splitter = lambda p: float(p.stem.split("_")[0])
    wavelengths, means = io.load_means(folder, selection=center, retrieve_value=splitter)

    # Remove images with (nearly) saturated pixels
    saturated = np.where(means >= 0.95 * camera.saturation)
    images_with_saturation = np.unique(saturated[0])
    means = np.delete(means, images_with_saturation, axis=0)
    wavelengths = np.delete(wavelengths, images_with_saturation)

    # Bias correction
    means = camera.correct_bias(means, selection=center)

    # Flat-field correction
    if flatfield:
        try:
            means = camera.correct_flatfield(means, selection=center)
        except AssertionError as e:
            print("Could not do flat-field correction, see below for error", e, sep="\n")

    # Demosaick the data
    means_RGBG2 = np.array(camera.demosaick(means, selection=center))

    # Get the mean per wavelength per channel and the standard deviations
    means_final = np.nanmean(means_RGBG2, axis=(2,3))
    stds_final = np.nanstd(means_RGBG2, axis=(2,3))
    print("\n...Finished!")

    return wavelengths, means_final, stds_final, means_RGBG2


def load_monochromator_data_multiple(camera, folders, **kwargs):
    """
    Wrapper around `load_monochromator_data` that does multiple files. Ensures
    the outputs are in a convenient format.
    """
    # First, load all the data
    data = [load_monochromator_data(camera, folder, **kwargs) for folder in folders]

    # Then split out all the constituents
    wavelengths, means, stds, means_RGBG2 = zip(*data)

    # Make labels for each data set
    labels = [j*np.ones_like(wvl).astype(np.uint8) for j, wvl in enumerate(wavelengths)]

    # Now return everything
    return wavelengths, means, stds, means_RGBG2, labels


def generate_slices_for_RGBG2_bands(slice_length, nr_bands=4):
    """
    Generate slices for data that have been flattened along the RGBG2 (band) axis.
    Mostly to be used as a helper for flatten_monochromator_image_data.
    """
    slices = [np.s_[slice_length*j:slice_length*(j+1)] for j in range(nr_bands)]

    return slices


def adjust_slices_for_RGBG2_bands_multi(RGBG2_slices, nr_bands=4):
    """
    Adjust slices for data that have been flattened along the RGBG2 (band) axis.
    Make the slices for each data set start at the end of the previous data set, rather than 0.
    """
    # Generate the slices with 0 as the start
    RGBG2_slices = np.array(RGBG2_slices)

    # Loop over the slices and add the end of the previous set to the start of the next one
    for j, _ in enumerate(RGBG2_slices[1:], start=1):
        offset = RGBG2_slices[j-1,-1].stop
        for i in range(nr_bands):
            old = RGBG2_slices[j,i]
            RGBG2_slices[j,i] = slice(old.start+offset, old.stop+offset, None)

    return RGBG2_slices


def flatten_monochromator_image_data(means_RGBG2):
    """
    Flatten the mean image data from a monochromator.
    Original data typically have the shape
    [nr wavelengths, nr bands, size x, size y]
    This function flattens them to
    [nr wavelengths * nr bands, all pixels]
    Note that the axis is band-first, then wavelength. This means the data look
    like (R1, R2, R3, ..., G1, G2, ...).

    Additional output includes slices to select the individual bands.
    """
    # Get the size of the first two axes and their product
    nr_wavelengths, nr_bands = means_RGBG2.shape[:2]
    spectral_axis_length = nr_wavelengths * nr_bands

    # First remove the spatial information
    means_flattened = np.reshape(means_RGBG2, (nr_wavelengths, nr_bands, -1))
    # Then swap the wavelength and filter axes
    means_flattened = np.swapaxes(means_flattened, 0, 1)
    # Finally, flatten the array further
    means_flattened = np.reshape(means_flattened, (spectral_axis_length, -1))

    # Slices to select R, G, B, and G2
    RGBG2_slices = generate_slices_for_RGBG2_bands(nr_wavelengths, nr_bands)

    return means_flattened, RGBG2_slices


def flatten_monochromator_image_data_multiple(means_RGBG2, *properties, nr_bands=4):
    """
    Flatten the mean image data from a monochromator.
    Handles multiple data sets.

    Other properties, such as wavelengths or labels, may also be passed.
    These will then be sorted and reshaped similarly.
    """
    # Flatten the main data first
    means_flattened, RGBG2_slices = zip(*[flatten_monochromator_image_data(means) for means in means_RGBG2])
    means_flattened = np.concatenate(means_flattened)

    # Because the slices are generated individually for each array, we need to update them
    # The slices for each data set should start at the end of the previous data set, rather than 0
    RGBG2_slices = adjust_slices_for_RGBG2_bands_multi(RGBG2_slices)

    # Now flatten the other properties
    properties = [np.concatenate([np.tile(p, nr_bands) for p in prop]) for prop in properties]

    return means_flattened, RGBG2_slices, *properties
