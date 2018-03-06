from matplotlib import pyplot as plt
import numpy as np
import rawpy
from cv2 import resize
from scipy.ndimage.filters import gaussian_filter1d as gauss
from sys import argv
from scipy.optimize import curve_fit

filename = argv[1]

TLpeaks = np.array([436.6, 487.7, 544.45, 611.6])

col0 = 1530
col1 = 1911
col2 = 1970
col3 = 2315

row0 = 2200
row1 = 3700
x = np.arange(row0, row1)

img = rawpy.imread(filename)
data = img.postprocess(use_camera_wb=True, gamma=(1,1), output_bps=8)

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
    plt.show()

def rgbplot(x, y, func=plt.plot):
    RGB = ["R", "G", "B"]
    for j in (0,1,2):
        func(x, y[..., j], c=RGB[j])

for D, DF in zip([thick, thin], [thickF, thinF]):
    for d in (D, DF):
        rgbplot(x, d[:, 100])
        plt.xlim(row0, row1)
        plt.ylim(0,255)
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

rgbplot(y, p)
plt.show()
rgbplot(y, p_fit)
plt.show()

p_diff = p - p_fit
print(f"Mean difference {p_diff.mean():.1f}, STD {p_diff.std():.1f}")

coeffarr = np.tile(np.nan, (y.shape[0], 2))
wvlfit = np.tile(np.nan, (y.shape[0], 3))
for i, col in enumerate(y):
    coeffarr[i] = np.polyfit(p_fit[i], TLpeaks[[3,2,0]], 1)
    wvlfit[i] = np.polyval(coeffarr[i], p_fit[i])

x2 = (TLpeaks[2]-coeffarr[:,1])/coeffarr[:,0]
x3 = (TLpeaks[3]-coeffarr[:,1])/coeffarr[:,0]
for D, ex in zip([thick, thin], [(col0, col1), (col2, col3)]):
    plt.imshow(D.astype("uint8"), extent=(*ex, row1, row0))
    plt.plot(y, x2, ls="--", c="w")
    plt.plot(y, x3, ls="--", c="w")
    plt.xlim(*ex)
    plt.show()

aco = np.polyfit(y, coeffarr[:,0], 2)
afit = np.polyval(aco, y)
bco = np.polyfit(y, coeffarr[:,1], 2)
bfit = np.polyval(bco, y)
plt.plot(y, coeffarr[:,0], c='r')
plt.plot(y, afit, c='k')
plt.show()
plt.plot(y, coeffarr[:,1], c='r')
plt.plot(y, bfit, c='k')
plt.show()

def wavelength_fit(y, a_coeff, b_coeff):
    a = np.polyval(a_coeff, y)
    b = np.polyval(b_coeff, y)
    def wavelength(x):
        return a*x + b
    return wavelength

column = 1700
lam = wavelength_fit(column, aco, bco)(x)
rgbplot(lam, thickF[:, column-col0])
plt.xlim(370, 740)
plt.ylim(0, 255)
plt.show()

def interpolate(wavelengths, rgb, lamrange):
    interpolated = np.vstack([np.interp(lamrange, wavelengths, rgb[:,j]) for j in (0,1,2)]).T
    return interpolated

def stack(column0, rgb, a_coeff, b_coeff, lamrange = np.arange(370, 740, 0.25)):
    wavelength_funcs = [wavelength_fit(c, a_coeff, b_coeff) for c in range(column0, column0+rgb.shape[1])]
    interpolated = np.array([interpolate(wavelength_funcs[i](x), rgb[:,i], lamrange) for i in range(rgb.shape[1])])
    mean = interpolated.mean(axis=0)
    return lamrange, mean

for D, c in zip([thickF, thinF], [col0, col2]):
    wavelength, intensity = stack(c, D, aco, bco)
    rgbplot(wavelength, intensity)
    plt.xlim(370, 740)
    plt.ylim(0,255)
    plt.show()