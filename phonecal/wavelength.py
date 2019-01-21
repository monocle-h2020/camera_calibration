from .general import x_spectrum, y_thick, y_thin, y
from . import raw
from astropy.stats import sigma_clip

import numpy as np

fluorescent_lines = np.array([611.6, 544.45, 436.6])  # RGB, units: nm
degree_of_spectral_line_fit = 2
degree_of_wavelength_fit = 2
degree_of_coefficient_fit = 4
wavelength_limits = (350, 750)

def find_RGB_lines(image, offset=0):
    return image.argmax(axis=0) + offset

def find_fluorescent_lines_old(thick, thin, offset=x_spectrum[0]):
    lines_thick = find_RGB_lines(thick, offset=offset)
    lines_thin  = find_RGB_lines(thin , offset=offset)
    l = np.concatenate((lines_thick, lines_thin))
    l_fit = l.copy()
    for j in (0,1,2):  # fit separately for R, G, B
        coeff = np.polyfit(y, l[:,j], degree_of_spectral_line_fit)
        l_fit[:,j] = np.polyval(coeff, y)

    return l, l_fit

def find_fluorescent_lines(RGB):
    RGB_copy = RGB.copy()
    RGB_copy[np.isnan(RGB_copy)] = -999
    peaks = np.nanargmax(RGB_copy, axis=2).astype(np.float32)
    peaks[peaks == 0] = np.nan
    return peaks

def fit_fluorescent_lines(lines):
    lines_fit = lines.copy()
    for j in (0,1,2):  # fit separately for R, G, B
        idx = np.isfinite(lines[j])
        new_y = raw.y[idx] ; new_line = lines[j][idx]
        clipped = sigma_clip(new_line)  # generates a masked array
        idx = ~clipped.mask  # get the non-masked items
        new_y = new_y[idx] ; new_line = new_line[idx]
        # np.polyfit can go along axis - try this?
        coeff = np.polyfit(new_y, new_line, degree_of_spectral_line_fit)
        lines_fit[j] = np.polyval(coeff, raw.y)
    return lines_fit

def fit_single_wavelength_relation(lines):
    coeffs = np.polyfit(lines, fluorescent_lines, degree_of_wavelength_fit)
    return coeffs

def fit_many_wavelength_relations(y, lines):
    coeffarr = np.tile(np.nan, (y.shape[0], degree_of_wavelength_fit+1))
    for i, col in enumerate(y):
        coeffarr[i] = fit_single_wavelength_relation(lines[:,i])

    return coeffarr

def fit_wavelength_coefficients(y, coefficients):
    coeff_coeff = np.array([np.polyfit(y, coefficients[:, i], degree_of_coefficient_fit) for i in range(degree_of_wavelength_fit+1)])
    coeff_fit = np.array([np.polyval(coeff, y) for coeff in coeff_coeff]).T
    return coeff_coeff, coeff_fit

def wavelength_fit(y, *coeff_coeff):
    coeff = [np.polyval(co, y) for co in coeff_coeff]
    def wavelength(x):
        return np.polyval(coeff, x)
    return wavelength

def calculate_wavelengths(coeff, x, y):
    coeff_fit = np.array([np.polyval(c, y) for c in coeff]).T
    wavelengths = np.array([np.polyval(c_fit, x) for c_fit in coeff_fit])
    return wavelengths

def interpolate_old(wavelengths, rgb, lambdarange):
    interpolated = np.vstack([np.interp(lambdarange, wavelengths, rgb[:,j]) for j in (0,1,2)]).T
    return interpolated

def interpolate(wavelength_array, color_value_array, lambdarange):
    interpolated = np.array([np.interp(lambdarange, wavelengths, color_values)
    for wavelengths, color_values in zip(wavelength_array, color_value_array)])
    return interpolated

def interpolate_multi(wavelengths_split, RGBG, lambdamin=340, lambdamax=760, lambdastep=0.5):
    lambdarange = np.arange(lambdamin, lambdamax+lambdastep, lambdastep)
    all_interpolated = np.array([interpolate(wavelengths_split[c], RGBG[c], lambdarange) for c in range(4)])
    all_interpolated = np.moveaxis(all_interpolated, 2, 1)
    return lambdarange, all_interpolated

def stack(wavelengths, interpolated):
    stacked = interpolated.mean(axis=2)
    stacked = np.roll(stacked, 1, axis=0)  # move to make space for wavelengths
    stacked[2] = (stacked[0] + stacked[2])/2.  # G becomes mean of G
    stacked[0] = wavelengths  # put wavelengths into array
    return stacked

def per_wavelength(wavelengths_split, RGBG):
    nm_diff = np.diff(wavelengths_split, axis=1)/2
    nm_width = nm_diff[:,1:,:] + nm_diff[:,:-1,:]
    wavelengths_split_new = wavelengths_split[:,1:-1,:]
    RGBG_new = RGBG[:,1:-1,:] / nm_width
    return wavelengths_split_new, RGBG_new

def resolution(wavelengths, intensity, limit=0.5):
    max_px = intensity.argmax()
    max_in = intensity.max()
    half_right = np.where(intensity[max_px:] < max_in*limit)[0][0] + max_px
    half_left  = max_px - np.where(intensity[max_px::-1] < max_in*limit)[0][0]
    return wavelengths[half_right] - wavelengths[half_left]

def save_coefficients(coefficients, saveto="wavelength.npy"):
    np.save(saveto, coefficients)

def load_coefficients(filename="wavelength.npy"):
    coefficients = np.load(filename)
    return coefficients