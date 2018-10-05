import numpy as np
import rawpy
from sys import argv
from matplotlib import pyplot as plt, patheffects as pe
from phonecal.general import gauss_filter
from phonecal import raw, plot, io, wavelength

LUT = np.load("results/gain_new/LUT.npy")

filename = argv[1]
handle = "_".join(filename.split("/")).split(".")[0]

ISO = int(argv[2])  # temporary to test Gain
texp_inv = int(argv[3])  # temporary to test Gain

img = io.load_dng_raw(filename)
exif = io.load_exif(filename)
image_cut  = raw.cut_out_spectrum(img.raw_image)
colors_cut = raw.cut_out_spectrum(img.raw_colors)

RGBG, offsets = raw.pull_apart(image_cut, colors_cut)
plot.RGBG_stacked(RGBG, extent=(raw.xmin, raw.xmax, raw.ymax, raw.ymin), show_axes=True, saveto="RGBG_cutout.png")

coefficients = wavelength.load_coefficients("wavelength_solution.npy")
wavelengths_cut = wavelength.calculate_wavelengths(coefficients, raw.x, raw.y)
wavelengths_split, offsets = raw.pull_apart(wavelengths_cut, colors_cut)

lambdarange, all_interpolated = wavelength.interpolate_multi(wavelengths_split, RGBG)
plot.RGBG_stacked(all_interpolated.swapaxes(1,2), extent=(lambdarange[0], lambdarange[-1], raw.ymax, raw.ymin), show_axes=True, xlabel="$\lambda$ (nm)", aspect=0.5 * len(lambdarange) / len(raw.x), saveto="RGBG_cutout_corrected.png")
plot.RGBG_stacked_with_graph(all_interpolated, x=lambdarange, extent=(lambdarange[0], lambdarange[-1], raw.ymax, raw.ymin), xlabel="$\lambda$ (nm)", aspect=0.5 * len(lambdarange) / len(raw.x), saveto="RGBG_cutout_corrected_spectrum.png")

plot.RGBG(RGBG, vmax=800, saveto="RGBG_split.png", size=30)
plot.RGBG(all_interpolated.swapaxes(1,2), vmax=800, saveto="RGBG_split_interpolated.png", size=30)

stacked = wavelength.stack(lambdarange, all_interpolated)
stacked[1:] -= 528
np.save("results/spectra/"+handle+"_spectrum.npy", stacked)

plot.plot_spectrum(stacked[0], stacked[1:], saveto="results/spectra/"+handle+"_spectrum_rgb.png",
                   xlim=(340, 760), ylim=(0, None), title=exif["Image DateTime"].values)

gain = LUT[1, ISO]
stacked[1:] *= gain * texp_inv  # convert ADU to e-/s
plot.plot_spectrum(stacked[0], stacked[1:], xlim=(340, 760), ylim=(0, 70), title=f"ISO {ISO} ; 1/t_exp {texp_inv:.1f}",
                   ylabel="$e^-/s$", saveto=f"ISO_{ISO}_texp_{texp_inv:.1f}.png")
