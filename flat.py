import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import raw, io, plot
from phonecal.general import gaussMd, distances_px, Rsquare, curve_fit, RMS

meanfile = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(meanfile)
phone = io.read_json(root/"info.json")

iso = io.split_iso(meanfile)
print("Loaded information")

bias = phone["software"]["bias"]

stdsfile = meanfile.parent / meanfile.name.replace("mean", "stds")

mean = np.load(meanfile)
stds = np.load(stdsfile)
colours = io.load_colour(stacks)

mean -= bias

mRGBG, offsets = raw.pull_apart(mean, colours)
sRGBG, offsets = raw.pull_apart(stds, colours)

# rescale to normalised values
normalisation = mRGBG.max(axis=(1,2))[:,np.newaxis,np.newaxis]
mRGBG = mRGBG / normalisation
sRGBG = sRGBG / normalisation

flat_field = raw.put_together_from_colours(mRGBG, colours)
flat_field_gauss = gaussMd(flat_field, 10)
np.save(products/"flatfield.npy", flat_field_gauss)
print("Saved array")

vmin, vmax = np.nanmin(mRGBG), 1
plot.show_RGBG(mRGBG, colorbar_label=25*" "+"Relative sensitivity", vmin=vmin, vmax=1, saveto=results/f"flat/iso{iso}.pdf")
print("Made RGBG images")

print("Minima:", mRGBG.min(axis=(1,2)))

plt.figure(figsize=(3,2), tight_layout=True)
img = plt.imshow(mRGBG[0], cmap=plot.cmaps["Rr"])
plt.xticks([])
plt.yticks([])
colorbar_here = plot.colorbar(img)
colorbar_here.set_label("Relative sensitivity")
colorbar_here.locator = plot.ticker.MaxNLocator(nbins=4)
colorbar_here.update_ticks()
plt.savefig(results/f"flat/iso{iso}_R.pdf")
plt.show()
plt.close()
print("Made single plot")

plt.figure(figsize=(5,5), tight_layout=True)
img = plt.imshow(flat_field)
plt.xticks([])
plt.yticks([])
colorbar_here = plot.colorbar(img)
colorbar_here.set_label("Relative sensitivity")
colorbar_here.locator = plot.ticker.MaxNLocator(nbins=4)
colorbar_here.update_ticks()
plt.savefig(results/f"flat/iso{iso}_full.pdf")
plt.show()
plt.close()
print("Made full plot")

mid1, mid2 = mRGBG.shape[1]//2, mRGBG.shape[2]//2
x = np.arange(0, mean.shape[0], 2)
y = np.arange(0, mean.shape[1], 2)
plt.plot(y, mRGBG[0,mid1], c='k')
plt.fill_between(y, mRGBG[0,mid1]-sRGBG[0,mid1], mRGBG[0,mid1]+sRGBG[0,mid1], color="0.5")
plt.xlabel("Y position")
plt.ylabel("Relative sensitivity")
plt.xlim(0, mean.shape[1])
plt.ylim(0, 1.1)
plt.grid()
plt.savefig(results/f"flat/iso{iso}_vertical.pdf")
plt.show()
plt.close()

plt.plot(x, mRGBG[0,:,mid2], c='k')
plt.fill_between(x, mRGBG[0,:,mid2]-sRGBG[0,:,mid2], mRGBG[0,:,mid2]+sRGBG[0,:,mid2], color="0.5")
plt.xlabel("X position")
plt.ylabel("Relative sensitivity")
plt.xlim(0, mean.shape[0])
plt.ylim(0, 1.1)
plt.grid()
plt.savefig(results/f"flat/iso{iso}_horizontal.pdf")
plt.show()
plt.close()

X, Y, D = distances_px(flat_field)

XY = np.stack([X.ravel(), Y.ravel()])

def vignette_radial(XY, k0, k1, k2, k3, k4, cx_hat, cy_hat):
    """
    Vignetting function as defined in Adobe DNG standard 1.4.0.0

    Parameters
    ----------
    XY
        array with X and Y positions of pixels, in absolute (pixel) units
    k0, ..., k4
        polynomial coefficients
    cx_hat, cy_hat
        optical center of image, in normalized euclidean units (0-1)
        relative to the top left corner of the image
    """
    x, y = XY

    x0, y0 = x[0], y[0] # top left corner
    x1, y1 = x[-1], y[-1]  # bottom right corner
    cx = x0 + cx_hat * (x1 - x0)
    cy = y0 + cy_hat * (y1 - y0)
    # (cx, cy) is the optical center in absolute (pixel) units
    mx = max([abs(x0 - cx), abs(x1 - cx)])
    my = max([abs(y0 - cy), abs(y1 - cy)])
    m = np.sqrt(mx**2 + my**2)
    # m is the euclidean distance from the optical center to the farthest corner in absolute (pixel) units
    r = 1/m * np.sqrt((x - cx)**2 + (y - cy)**2)
    # r is the normalized euclidean distance of every pixel from the optical center (0-1)

    p = [k4, 0, k3, 0, k2, 0, k1, 0, k0, 0, 1]
    g = np.polyval(p, r)
    # g is the normalization factor to multiply measured values with

    return g

correction = 1 / flat_field

plt.figure(figsize=(5,5), tight_layout=True)
img = plt.imshow(correction, vmin=1, vmax=correction.max())
plt.xticks([])
plt.yticks([])
colorbar_here = plot.colorbar(img)
colorbar_here.set_label("Correction factor (observed)")
colorbar_here.update_ticks()
plt.savefig(results/f"flat/iso{iso}_correction_observed.pdf")
plt.show()
plt.close()

popt, pcov = curve_fit(vignette_radial, XY, correction.ravel(), p0=[1, 2, -5, 5, -2, 0.5, 0.5])
standard_errors = np.sqrt(np.diag(pcov))

print("Parameter +- Error    ; Relative error")
for p, s in zip(popt, standard_errors):
    print(f"{p:+.6f} +- {s:.6f} ; {abs(100*s/p):.3f} %")

g_fit = vignette_radial(XY, *popt).reshape(correction.shape)

plt.figure(figsize=(5,5), tight_layout=True)
img = plt.imshow(g_fit, vmin=1, vmax=correction.max())
plt.xticks([])
plt.yticks([])
colorbar_here = plot.colorbar(img)
colorbar_here.set_label("Correction factor (best fit)")
colorbar_here.update_ticks()
plt.savefig(results/f"flat/iso{iso}_correction_fit.pdf")
plt.show()
plt.close()

difference = correction - g_fit

plt.figure(figsize=(5,5), tight_layout=True)
img = plt.imshow(difference)
plt.xticks([])
plt.yticks([])
colorbar_here = plot.colorbar(img)
colorbar_here.set_label("Correction factor (observed - fit)")
colorbar_here.update_ticks()
plt.savefig(results/f"flat/iso{iso}_correction_difference.pdf")
plt.show()
plt.close()

plt.figure(figsize=(5,5), tight_layout=True)
plt.hist(difference.ravel(), bins=250)
plt.xlabel("Correction factor (observed - fit)")
plt.savefig(results/f"flat/iso{iso}_correction_difference_hist.pdf")
plt.show()
plt.close()

plt.figure(figsize=(5,5), tight_layout=True)
plt.hist(difference.ravel() / correction.ravel(), bins=250)
plt.xlabel("Correction factor (observed - fit)/observed")
plt.savefig(results/f"flat/iso{iso}_correction_difference_hist_relative.pdf")
plt.show()
plt.close()

print(f"RMS difference: {RMS(difference):.3f}")
print(f"RMS difference (relative): {100*RMS(difference/correction):.1f} %")
