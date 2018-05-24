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

def cut(arr, x=250, y=250):
    return arr[y:-y, x:-x]

filename = argv[1]
handle = filename.split("/")[-1].split(".")[0]
polarisation_angle = float(argv[2])
pixel_angle = np.load("pixel_angle.npy")
nr_points = 12500

img = io.load_dng_raw(filename)
image_cut  = cut(img.raw_image)
colors_cut = cut(img.raw_colors)

RGBG, offsets = raw.pull_apart(image_cut, colors_cut)
plot.RGBG_stacked(RGBG, show_axes=True, boost=1)

x = np.arange(image_cut.shape[1])
y = np.arange(image_cut.shape[0])
X, Y = np.meshgrid(x, y)
(x0, y0) = (len(x) / 2, len(y) / 2)
D = np.sqrt((X - x0)**2 + (Y - y0)**2)
D_split, offsets = raw.pull_apart(D, colors_cut)

#RGBG_norm = RGBG.astype(np.float) - 528
#RGBG_norm[RGBG_norm > 60000] = 0
#for j in range(4):
#    RGBG_norm[..., j] = RGBG_norm[..., j] / find_I0(RGBG_norm[..., j], D_split[..., j])

R, G, B, G2 = [RGBG[..., j].ravel() for j in range(4)]
DR, DG, DB, DG2 = [D_split[..., j].ravel() for j in range(4)]
G = np.concatenate((G, G2))
DG = np.concatenate((DG, DG2))
RGB = [R, G, B]
DRGB = [DR, DG, DB]

I_range = np.linspace(0, 1, 250)

colours = "RGB"
for colour, C, D in zip(colours, RGB, DRGB):
    st = int(len(D)/nr_points)
    o = 69
    plt.hist(C, bins=np.arange(0, 5000, 100), color=colour)
    plt.xlim(0, 4096)
    plt.xlabel(colour+" value")
    plt.ylabel("# pixels")
    plt.show()
    plt.close()

    d_range = np.arange(0, D.max(), 10)

    I0 = find_I0(C-528, D)

    plt.scatter(D[o::st], C[o::st], c=colour, s=3, label="Data")
    plt.plot(d_range, cos4f(d_range, pixel_angle, I0)+528, c='k', label="$\cos^4$ fit")
    plt.xlabel("Distance from center (px)")
    plt.ylabel(colour+" value")
    plt.legend(loc="lower left")
    plt.show()
    plt.close()

    I = cos4f(D, pixel_angle, malus(polarisation_angle))

    plt.scatter(I[o::st], C[o::st], c=colour, s=3)
    plt.xlabel("Intensity")
    plt.ylabel("RGB value")
    plt.xlim(0, 1)
    plt.ylim(0, 4096)
    plt.title(f"{colour} linearity ({len(I[o::st])/len(I)*100:.2f}% of data shown)")
    plt.show()
    plt.close()

    mean_per_I, bin_edges, bin_number = binned_statistic(I, C, statistic="mean",  bins=I_range)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges + bin_width/2
    std_per_I  = binned_statistic(I, C, statistic=np.std,  bins=I_range).statistic
    nr_per_I   = binned_statistic(I, C, statistic="count", bins=I_range).statistic
    idx = np.where(nr_per_I > 10e3)
    bin_centers = bin_centers[idx]
    mean_per_I = mean_per_I[idx]
    std_per_I = std_per_I[idx]
    nr_per_I = nr_per_I[idx]
    plt.scatter(bin_centers, mean_per_I, c=colour)
    plt.xlim(0,1)
    plt.ylim(0, 4096)
    plt.xlabel("Intensity")
    plt.ylabel("RGB value (binned)")
    plt.show()
    plt.close()

    plt.scatter(bin_centers, std_per_I**2./(mean_per_I - 528), c=colour)
    plt.xlim(0, 1)
    plt.xlabel("Intensity")
    plt.ylabel(f"$\sigma^2 / ({colour} - 528)$")
    plt.show()
    plt.close()

    plt.scatter(mean_per_I, std_per_I**2., c=colour)
    plt.xlim(0, 4096)
    plt.xlabel(f"{colour}")
    plt.ylabel("$\sigma^2$")
    plt.show()
    plt.close()

    np.save(f"gamma/{colour}_{handle}.npy", np.vstack((bin_centers, mean_per_I, std_per_I, nr_per_I)))