import numpy as np

fluorescent_lines = np.array([611.6, 544.45, 436.6])
degree_of_spectral_line_fit = 2
degree_of_wavelength_fit = 2
degree_of_coefficient_fit = 2
wavelength_limits = (350, 750)

def find_RGB_lines(image, offset=0):
    return image.argmax(axis=0) + offset

def find_fluorescent_lines(thick, thin, columns, offset=0):
    lines_thick = find_RGB_lines(thick, offset=offset)
    lines_thin  = find_RGB_lines(thin , offset=offset)
    y = np.concatenate((np.arange(columns[0], columns[1]), np.arange(columns[2], columns[3])))
    l = np.concatenate((lines_thick, lines_thin))
    l_fit = l.copy()
    for j in (0,1,2):  # fit separately for R, G, B
        coeff = np.polyfit(y, l[:,j], degree_of_spectral_line_fit)
        l_fit[:,j] = np.polyval(coeff, y)

    return y, l, l_fit

def fit_single_wavelength_relation(lines):
    coeffs = np.polyfit(lines, fluorescent_lines, degree_of_wavelength_fit)
    return coeffs

def fit_many_wavelength_relations(y, lines):
    coeffarr = np.tile(np.nan, (y.shape[0], degree_of_wavelength_fit+1))
    for i, col in enumerate(y):
        coeffarr[i] = fit_single_wavelength_relation(lines[i])

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

def interpolate(wavelengths, rgb, lambdarange):
    interpolated = np.vstack([np.interp(lambdarange, wavelengths, rgb[:,j]) for j in (0,1,2)]).T
    return interpolated

def stack(x, rgb, coeff, yoffset=0, lambdarange = np.arange(*wavelength_limits, 0.25)):
    wavelength_funcs = [wavelength_fit(c, *coeff) for c in range(yoffset, yoffset+rgb.shape[1])]
    wavelengths = np.array([f(x) for f in wavelength_funcs])
    # divide by nm/px to get intensity per nm
    rgb_new = rgb[:-1,:,:] / np.diff(wavelengths, axis=1).T[:,:,np.newaxis]
    interpolated = np.array([interpolate(wavelengths[i,:-1], rgb_new[:,i], lambdarange) for i in range(rgb.shape[1])])
    means = interpolated.mean(axis=0)
    return lambdarange, means

def resolution(wavelengths, intensity):
    max_px = intensity.argmax()
    max_in = intensity.max()
    half_right = np.where(intensity[max_px:] < max_in/2.)[0][0] + max_px
    half_left  = max_px - np.where(intensity[max_px::-1] < max_in/2.)[0][0]
    return wavelengths[half_right] - wavelengths[half_left]