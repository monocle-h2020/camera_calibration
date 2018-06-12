import numpy as np
import rawpy
from sys import argv
from matplotlib import pyplot as plt, patheffects as pe
from ispex.general import gauss_filter
from ispex import raw, plot, io, wavelength
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic

def malus(angle):
    return (np.cos(np.radians(angle)))**2

def cos4(d, p, a):
    return a*(np.cos(d/p))**4

def cos4f(d, f, a):
    return a*(np.cos(np.arctan(d/f)))**4

def find_I0(rgbg, distances, radius=100):
    return rgbg[distances < radius].mean()

def find_I1500(rgbg, distances):
    return rgbg[np.where((distances > 1500) & (distances < 1510))].mean()

def cut(arr, x=250, y=250):
    return arr[y:-y, x:-x]

I_range = np.linspace(0, 1, 251)

files, visual_angles = np.loadtxt(argv[1], dtype="S8", unpack=True, skiprows=1)
visual_angles = visual_angles.astype(float)

I1500s = np.zeros((3, len(files)))

for i, file in enumerate(files):
    filename = f"test_files/gamma/{file.decode()}.DNG"
    img = io.load_dng_raw(filename)
    image_cut  = cut(img.raw_image)
    colors_cut = cut(img.raw_colors)

    RGBG, offsets = raw.pull_apart(image_cut, colors_cut)
    print(file)
    plot.RGBG_stacked(RGBG, show_axes=True, boost=1)

    x = np.arange(image_cut.shape[1])
    y = np.arange(image_cut.shape[0])
    X, Y = np.meshgrid(x, y)
    (x0, y0) = (len(x) / 2, len(y) / 2)
    D = np.sqrt((X - x0)**2 + (Y - y0)**2)
    D_split, offsets = raw.pull_apart(D, colors_cut)

    R, G, B, G2 = [RGBG[..., j].ravel() for j in range(4)]
    DR, DG, DB, DG2 = [D_split[..., j].ravel() for j in range(4)]
    G = np.concatenate((G, G2))
    DG = np.concatenate((DG, DG2))
    RGB = [R, G, B]
    DRGB = [DR, DG, DB]

    colours = "RGB"
    for j, C, D in zip(range(3), RGB, DRGB):
        I1500 = find_I1500(C, D)
        I1500s[j, i] = I1500

def cos2(angle, amplitude, angle_offset, y_offset):
    angle_rad = np.radians(angle - angle_offset)
    cos2 = np.cos(angle_rad)**2.
    return y_offset + amplitude * cos2

idx = np.where(I1500s[2] > 540)
popt, pcov = curve_fit(cos2, visual_angles[idx], I1500s[2][idx], p0=[1500, 70, 2000])
polariser_angle = popt[1]

a = np.arange(0, 360, 3)
plt.figure(figsize=(10,5))
plt.scatter(visual_angles[idx], I1500s[2][idx], c='b', label="Data")
plt.plot(a, cos2(a, *popt), c='k', label=f"Fit ($\\alpha = {polariser_angle:.1f}$)")
plt.xlabel("Visual angle (degrees)")
plt.ylabel("Mean B value for $1500 < D < 1510$")
plt.legend(loc="best")
plt.show()
plt.close()
