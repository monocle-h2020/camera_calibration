import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from ispex.general import cut
from ispex.gamma import polariser_angle, I_range, cos4f, malus, find_I0, pixel_angle
from ispex import raw, plot, io
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic

filename = argv[1]
handle = filename.split("/")[-1].split(".")[0]
polarisation_angle = float(argv[2]) - polariser_angle
nr_points = 12500

img = io.load_dng_raw(filename)
image_cut  = cut(img.raw_image)
colors_cut = cut(img.raw_colors)

RGBG, offsets = raw.pull_apart(image_cut, colors_cut)
plot.RGBG_stacked(RGBG, show_axes=True, boost=1, saveto="A_plaatje.png")

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

def save_values(colour, I, RGB, bin_edges, saveto="gamma/"):
    for b, bplus in zip(bin_edges, bin_edges[1:]):
        filename = f"{saveto}/{colour}_{b:.3f}.npy"
        try:
            previous = np.load(filename)
        except FileNotFoundError:
            previous = np.array([])
        idx = np.where((I > b) & (I <= bplus))
        all_I = np.concatenate((previous, RGB[idx]))
        np.save(filename, all_I)

D_max = np.ceil(D_split.max()*1.01)

colours = "RGB"
for angle, colour, C, D in zip(pixel_angle, colours, RGB, DRGB):

    #idx = np.where((D > 250) & (D < 1850))
    #C = C[idx]
    #D = D[idx]

    plt.hist(C, bins=np.arange(0, 4200, 100), color=colour)
    plt.xlim(0, 4096)
    plt.xlabel(colour+" value")
    plt.ylabel("# pixels")
    plt.show()
    plt.close()

    d_range = np.arange(0, D_max, 15)

    I0 = C.max()

    cos = lambda distance, amp, offset: cos4f(distance, angle, amp, offset)
    popt, pcov = curve_fit(cos, D, C, p0=[I0, 528])

    plt.figure(figsize=(15,8))
    plot.hexbin_colour(colour, D, C, bins="log")
    plt.plot(d_range, cos(d_range, *popt), c='k', label="$\cos^4$ fit")
    plt.xlabel("Distance from center (px)")
    plt.xlim(0, D_max)
    plt.ylabel(colour+" value")
    plt.savefig(f"{colour}_cos4.png")
    plt.show()
    plt.close()

    I = cos4f(D, angle, malus(polarisation_angle), 0)

    plot.linearity(I, C, colour)

    mean_per_I, bin_edges, bin_number = binned_statistic(I, C, statistic="mean",  bins=I_range)
    bin_width = bin_edges[1] - bin_edges[0]
    bin_centers = bin_edges + bin_width/2
    std_per_I  = binned_statistic(I, C, statistic=np.std,  bins=I_range).statistic
    nr_per_I   = binned_statistic(I, C, statistic="count", bins=I_range).statistic
    idx = np.where(nr_per_I > 1e3)
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

    plt.scatter(bin_centers, std_per_I**2., c=colour)
    plt.xlim(0,1)
    plt.ylim(ymin=0)
    plt.xlabel("Intensity")
    plt.ylabel("$\sigma^2$")
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

    #save_values(colour, I, C, bin_edges)