import numpy as np
from matplotlib import pyplot as plt
from sys import argv
from phonecal import raw, plot, io, wavelength
from phonecal.general import blackbody, RMS, gauss1d

#wavelength, spectrometer = np.loadtxt("reference_spectra/sun.txt", skiprows=13, unpack=True)
#spectrometer /= spectrometer[wavelength == 500]
#ind = np.where((wavelength >= 390) & (wavelength <= 700))
#wavelength = wavelength[ind]
#spectrometer = spectrometer[ind]

wvl, smartsx, smartsy, smartsz = np.loadtxt("reference_spectra/ispex.ext.txt", skiprows=1, unpack=True)
smartsx /= smartsx[wvl == 500] ; smartsy /= smartsy[wvl == 500] ; smartsz /= smartsz[wvl == 500]
ind = np.where(wvl % 1 == 0)
wvl = wvl[ind]
smartsx = smartsx[ind] ; smartsy = smartsy[ind] ; smartsz = smartsz[ind]

BB = blackbody(wvl)
BB /= BB[wvl == 500]

#plt.plot(wavelength, spectrometer, c='b')
plt.plot(wvl, BB, c='k')
plt.plot(wvl, smartsx, c='r')
plt.xlim(390, 700)
plt.show()

file = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(file)
phone = io.read_json(root/"info.json")
coefficients = wavelength.load_coefficients(results/"ispex/wavelength_solution.npy")

img  = io.load_dng_raw(file)
exif = io.load_exif(file)

bias = phone["software"]["bias"]
values = img.raw_image.astype(np.float32) - bias

xmin, xmax = 2150, 3500
ymin_thin , ymax_thin  =  700, 1050
ymin_thick, ymax_thick = 1100, 1500
thin_slit  = np.s_[ymin_thin :ymax_thin , xmin:xmax]
thick_slit = np.s_[ymin_thick:ymax_thick, xmin:xmax]

x = np.arange(xmin, xmax)
y_thin = np.arange(ymin_thin, ymax_thin)
y_thick = np.arange(ymin_thick, ymax_thick)

image_thin   = values        [thin_slit ]
colors_thin  = img.raw_colors[thin_slit ]
RGBG_thin, _ = raw.pull_apart(image_thin, colors_thin)
plot.show_RGBG(RGBG_thin)

image_thick  = values        [thick_slit]
colors_thick = img.raw_colors[thick_slit]
RGBG_thick, _ = raw.pull_apart(image_thick, colors_thick)
plot.show_RGBG(RGBG_thick)

above_thin  = np.s_[ 590:600, xmin:xmax]
below_thick = np.s_[1570:1650, xmin:xmax]

values_above = values[above_thin]
colors_above = img.raw_colors[above_thin]
values_below = values[below_thick]
colors_below = img.raw_colors[below_thick]
RGBG_above,_ = raw.pull_apart(values_above, colors_above)
RGBG_below,_ = raw.pull_apart(values_below, colors_below)
above = RGBG_above.mean(axis=1)
below = RGBG_below.mean(axis=1)

RGBG_thin -= above[:,np.newaxis,:]
RGBG_thick -= below[:,np.newaxis,:]

wavelengths_thin  = wavelength.calculate_wavelengths(coefficients, x, y_thin )
wavelengths_thick = wavelength.calculate_wavelengths(coefficients, x, y_thick)
wavelengths_thin_RGBG , _ = raw.pull_apart(wavelengths_thin , colors_thin )
wavelengths_thick_RGBG, _ = raw.pull_apart(wavelengths_thick, colors_thick)

lambdarange, all_interpolated_thin  = wavelength.interpolate_multi(wavelengths_thin_RGBG , RGBG_thin )
lambdarange, all_interpolated_thick = wavelength.interpolate_multi(wavelengths_thick_RGBG, RGBG_thick)

stacked_thin  = wavelength.stack(lambdarange, all_interpolated_thin )
stacked_thick = wavelength.stack(lambdarange, all_interpolated_thick)

stacked_thin [1:] /= stacked_thin [1:].max(axis=1)[:,np.newaxis]
stacked_thick[1:] /= stacked_thick[1:].max(axis=1)[:,np.newaxis]

ind = np.where(lambdarange % 2 == 0)[0]
stacked_thin  = stacked_thin [:, ind]
stacked_thick = stacked_thick[:, ind]

BB = BB[ind]
smartsx = smartsx[ind]
wvl = wvl[ind]

smartsx_smooth = gauss1d(smartsx, 10)

def plot_spectral_response(wavelength, thin_spec, thick_spec, monochromator, title="", saveto=None):
    print(title)
    plt.figure(figsize=(7,3), tight_layout=True)
    for j, c in enumerate("rgb"):
        plt.plot(monochromator[0], monochromator[1+j], c=c)
        plt.plot(wavelength, thin_spec [1+j], c=c, ls="--")
        plt.plot(wavelength, thick_spec[1+j], c=c, ls="-.")
        print(f"{c}: thin: {RMS(monochromator[1+j] - thin_spec[1+j]):.2f}  ;  thick: {RMS(monochromator[1+j] - thick_spec[1+j]):.2f}")
    plt.plot([-10], [-10], c='k', label="Monochromator")
    plt.plot([-10], [-10], c='k', ls="--", label="iSPEX (narrow slit)")
    plt.plot([-10], [-10], c='k', ls="-.", label="iSPEX (broad slit)")
    plt.title(title)
    plt.xlim(390, 700)
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Relative sensitivity")
    plt.ylim(0, 1.01)
    plt.grid()
    plt.legend(loc="best")
    if saveto is not None:
        plt.savefig(saveto)
    plt.show()
    plt.close()

curves = np.load(results/"spectral_response/monochromator_curve.npy")

BB_thin  = stacked_thin  / BB
BB_thin  /= BB_thin .max(axis=1)[:,np.newaxis]
BB_thick = stacked_thick / BB
BB_thick /= BB_thick.max(axis=1)[:,np.newaxis]

plot_spectral_response(wvl, BB_thin, BB_thick, curves, "Black-body", saveto="results/ispex_black_body.pdf")

SMARTS_thin = stacked_thin / smartsx_smooth
SMARTS_thin /= SMARTS_thin.max(axis=1)[:,np.newaxis]
SMARTS_thick = stacked_thick / smartsx_smooth
SMARTS_thick /= SMARTS_thick.max(axis=1)[:,np.newaxis]

plot_spectral_response(wvl, SMARTS_thin, SMARTS_thick, curves, "SMARTS2", saveto="results/ispex_smarts2.pdf")
