from matplotlib import pyplot as plt
import numpy as np
import rawpy
from scipy.ndimage.filters import gaussian_filter1d as gauss
from sys import argv
from scipy.optimize import curve_fit
from exifread import process_file

filename = argv[1]

TLpeaks = np.array([436.6, 487.7, 544.45, 611.6])

degree = 2

col0 = 1530
col1 = 1911
col2 = 1970
col3 = 2315

row0 = 2150
row1 = 3900
x = np.arange(row0, row1)

lam0 = 350
lam1 = 750

img = rawpy.imread(filename)
data = img.postprocess(use_camera_wb=True, gamma=(1,1), output_bps=8)

with open(filename, "rb") as f:
    exif = process_file(f)

thick = data[row0:row1, col0:col1]
thin  = data[row0:row1, col2:col3]

for D, ex in zip([thick, thin], [(col0, col1), (col2, col3)]):
    plt.imshow(D, extent=(*ex, row1, row0))
    plt.show()

def gauss_filter(D, sigma=7, *args, **kwargs):
    """
    Apply a 1-D Gaussian kernel along the wavelength axis
    """
    return gauss(D.astype(float), sigma, *args, axis=0, **kwargs)

thickF = gauss_filter(thick)
thinF  = gauss_filter(thin )

for D, ex in zip([thickF, thinF], [(col0, col1), (col2, col3)]):
    plt.imshow(D.astype("uint8"), extent=(*ex, row1, row0))
    plt.xlabel("$y$")
    plt.ylabel("$x$")
    plt.show()

def rgbplot(x, y, func=plt.plot, **kwargs):
    RGB = ["R", "G", "B"]
    for j in (0,1,2):
        func(x, y[..., j], c=RGB[j], **kwargs)

x_example = 100
for D, DF in zip([thick, thin], [thickF, thinF]):
    for d in (D, DF):
        rgbplot(x, d[:, x_example])
#        plt.plot(x, d[:,100,2]*d[:,100,1]/7, c='k')
        plt.xlim(row0, row1)
        plt.ylim(0,255)
        # adjust title to absolute column value!
        plt.title(f"RGB values at $x = {x_example}$")
        plt.xlabel("$y$")
        plt.ylabel("$C$")
        plt.show()

def find_3peaks(D, start=0):
    return D.argmax(axis=0) + start

peaks_thick = find_3peaks(thickF, start=row0)
peaks_thin = find_3peaks(thinF, start=row0)
y = np.concatenate((np.arange(col0, col1), np.arange(col2, col3)))
p = np.concatenate((peaks_thick, peaks_thin))
p_fit = p.copy()
for j in (0,1,2):
    coeff = np.polyfit(y, p[:,j], 2)
    p_fit[:,j] = np.polyval(coeff, y)

rgbplot(y, p, func=plt.scatter, alpha=0.03)
rgbplot(y, p_fit, ls="--")
plt.title("Locations of RGB maxima")
plt.xlabel("$y$")
plt.ylabel("$x_{peak}$%")
plt.show()

p_diff = p - p_fit
print(f"Mean difference in peak location: {p_diff.mean():.1f}, STD {p_diff.std():.1f}")

coeffarr = np.tile(np.nan, (y.shape[0], degree+1))
wvlfit = np.tile(np.nan, (y.shape[0], 3))
for i, col in enumerate(y):
    coeffarr[i] = np.polyfit(p_fit[i], TLpeaks[[3,2,0]], degree)
    wvlfit[i] = np.polyval(coeffarr[i], p_fit[i])

rgbplot(y, wvlfit-TLpeaks[[3,2,0]],func=plt.scatter)

#x2 = (TLpeaks[2]-coeffarr[:,1])/coeffarr[:,0]
#x3 = (TLpeaks[3]-coeffarr[:,1])/coeffarr[:,0]
#for D, ex in zip([thick, thin], [(col0, col1), (col2, col3)]):
#    plt.imshow(D.astype("uint8"), extent=(*ex, row1, row0))
#    plt.plot(y, x2, ls="--", c="w")
#    plt.plot(y, x3, ls="--", c="w")
#    plt.xlim(*ex)
#    plt.show()

coeff_coeff = np.array([np.polyfit(y, coeffarr[:, i], 2) for i in range(degree+1)])
coeff_fit = np.array([np.polyval(coeff, y) for coeff in coeff_coeff]).T
for i in range(degree+1):
    plt.scatter(y, coeffarr[:,i], c='r')
    plt.plot(y, coeff_fit[:,i], c='k', lw=3)
    plt.xlim(y[0], y[-1])
    plt.ylim(coeffarr[:,i].min(), coeffarr[:,i].max())
    plt.title(f"Coefficient {i} of wavelength fit")
    plt.xlabel("$y$")
    plt.ylabel(f"$p_{i}$")
    plt.show()

def wavelength_fit(y, *coeff_coeff):
    coeff = [np.polyval(co, y) for co in coeff_coeff]
    def wavelength(x):
        return np.polyval(coeff, x)
    return wavelength

def interpolate(wavelengths, rgb, lamrange):
    interpolated = np.vstack([np.interp(lamrange, wavelengths, rgb[:,j]) for j in (0,1,2)]).T
    return interpolated

def stack(column0, rgb, *coeff_coeff, lamrange = np.arange(lam0, lam1, 0.25)):
    wavelength_funcs = [wavelength_fit(c, *coeff_coeff) for c in range(column0, column0+rgb.shape[1])]
    wavelengths = np.array([f(x) for f in wavelength_funcs])
    rgb_new = rgb[:-1,:,:] / np.diff(wavelengths, axis=1).T[:,:,np.newaxis]
    interpolated = np.array([interpolate(wavelengths[i,:-1], rgb_new[:,i], lamrange) for i in range(rgb.shape[1])])
    mean = interpolated.mean(axis=0)
    return lamrange, mean

wavelength, intensity_thick = stack(col0, thickF, *coeff_coeff)
wavelength, intensity_thin  = stack(col2, thinF , *coeff_coeff)

for i in (intensity_thick, intensity_thin):
    rgbplot(wavelength, i)
    plt.xlim(lam0, lam1)
    plt.ylim(ymin=0)
    plt.title("Stacked RGB spectrum")
    plt.xlabel("$\lambda$ (nm)")
    plt.ylabel("C (a.u.)")
    plt.show()

def resolution(wavelength, intensity):
    max_px = intensity.argmax()
    max_in = intensity.max()
    half_right = np.where(intensity[max_px:] < max_in/2.)[0][0] + max_px
    half_left  = max_px - np.where(intensity[max_px::-1] < max_in/2.)[0][0]
    return wavelength[half_right] - wavelength[half_left]

for profile in [*intensity_thick.T, *intensity_thin.T]:
    res = resolution(wavelength, profile)
    print(f"Resolution: {res:.1f} nm")

wb = data[row0:row1, :col0-100].mean(axis=(0,1))
for i in (intensity_thick, intensity_thin):
    rgbplot(wavelength, i/wb)
    plt.xlim(lam0, lam1)
    plt.ylim(ymin=0)
    plt.title("Stacked RGB spectrum (post-WB)")
    plt.xlabel("$\lambda$ (nm)")
    plt.ylabel("C (a.u.)")
    plt.show()