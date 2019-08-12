import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import raw, io, plot, flat
from spectacle.general import gaussMd, Rsquare, RMS

meanfile = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(meanfile)
phone = io.read_json(root/"info.json")

label = meanfile.stem.split("_mean")[0]
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
normalisation = gaussMd(mRGBG, sigma=(0,5,5)).max(axis=(1,2))[:,np.newaxis,np.newaxis]
mRGBG = mRGBG / normalisation
sRGBG = sRGBG / normalisation

flat_field = raw.put_together_from_colours(mRGBG, colours)
flat_field_gauss = gaussMd(flat_field, 10)

# selection for further analysis
flat_field_gauss = flat_field_gauss[flat.clip_border]

vmin, vmax = np.nanmin(mRGBG), 1
plot.show_RGBG(mRGBG, colorbar_label=25*" "+"Relative sensitivity", vmin=vmin, vmax=1, saveto=results/f"flat/{label}.pdf")
print("Made RGBG images")

print("Minima:", mRGBG.min(axis=(1,2)))

plt.figure(figsize=(3,2), tight_layout=True)
img = plt.imshow(mRGBG[0], cmap=plot.cmaps["Rr"])
plt.xticks([])
plt.yticks([])
colorbar_here = plot.colorbar(img, location="right")
colorbar_here.set_label("Relative sensitivity")
colorbar_here.locator = plot.ticker.MaxNLocator(nbins=4)
colorbar_here.update_ticks()
plt.savefig(results/f"flat/{label}_R.pdf")
plt.show()
plt.close()
print("Made single plot of sensitivity")

plt.figure(figsize=(3,2), tight_layout=True)
img = plt.imshow(1/mRGBG[0], cmap=plot.cmaps["Rr"])
plt.xticks([])
plt.yticks([])
colorbar_here = plot.colorbar(img, location="right")
colorbar_here.set_label("Correction factor")
colorbar_here.locator = plot.ticker.MaxNLocator(nbins=4)
colorbar_here.update_ticks()
plt.savefig(results/f"flat/{label}_R_correction.pdf")
plt.show()
plt.close()
print("Made single plot of correction factor")

plt.figure(figsize=(5,5), tight_layout=True)
img = plt.imshow(flat_field)
plt.xticks([])
plt.yticks([])
colorbar_here = plot.colorbar(img)
colorbar_here.set_label("Relative sensitivity")
colorbar_here.locator = plot.ticker.MaxNLocator(nbins=4)
colorbar_here.update_ticks()
plt.savefig(results/f"flat/{label}_full.pdf")
plt.show()
plt.close()
print("Made full plot")

correction = 1 / flat_field_gauss
correction_raw = 1 / flat_field[flat.clip_border]
print(f"Maximum correction factor (raw): {correction_raw.max():.2f}")
print(f"Maximum correction factor (gauss): {correction.max():.2f}")

plt.figure(figsize=(5,5), tight_layout=True)
img = plt.imshow(correction, vmin=1, vmax=correction.max())
plt.xticks([])
plt.yticks([])
colorbar_here = plot.colorbar(img)
colorbar_here.set_label("Correction factor (observed)")
colorbar_here.update_ticks()
plt.savefig(results/f"flat/{label}_correction_observed.pdf")
plt.show()
plt.close()

popt, standard_errors = flat.fit_vignette_radial(correction)

np.save(products/f"flat_{label}_parameters.npy", np.stack([popt, standard_errors]))
np.save(products/f"flat_{label}_correction.npy", correction)
np.save(products/f"flat_{label}_correction_raw.npy", correction_raw)
print("Saved look-up table & parameters")

print("Parameter +- Error    ; Relative error")
for p, s in zip(popt, standard_errors):
    print(f"{p:+.6f} +- {s:.6f} ; {abs(100*s/p):.3f} %")

g_fit = flat.apply_vignette_radial(correction.shape, popt)

plt.figure(figsize=(5,5), tight_layout=True)
img = plt.imshow(g_fit, vmin=1, vmax=correction.max())
plt.xticks([])
plt.yticks([])
colorbar_here = plot.colorbar(img)
colorbar_here.set_label("Correction factor (best fit)")
colorbar_here.update_ticks()
plt.savefig(results/f"flat/{label}_correction_fit.pdf")
plt.show()
plt.close()

mid1, mid2 = np.array(correction.shape)//2
x = np.arange(0, correction.shape[0])
y = np.arange(0, correction.shape[1])

R2_y = Rsquare(correction[mid1], g_fit[mid1])
plt.scatter(y, correction[mid1], c='r', label="Observed")
plt.plot(y, g_fit[mid1], c='k', label="Fit")
plt.xlabel("Y position")
plt.ylabel("Correction factor")
plt.title(f"Row {mid1}: $R^2 = {R2_y:.3f}$")
plt.xlim(0, correction.shape[1])
plt.ylim(0.9, 1.03*correction[mid1].max())
plt.grid()
plt.legend(loc="upper center")
plt.savefig(results/f"flat/{label}_row.pdf")
plt.show()
plt.close()

R2_x = Rsquare(correction[:, mid2], g_fit[:, mid2])
plt.scatter(x, correction[:, mid2], c='r', label="Observed")
plt.plot(x, g_fit[:, mid2], c='k', label="Fit")
plt.xlabel("X position")
plt.ylabel("Correction factor")
plt.title(f"Column {mid2}: $R^2 = {R2_x:.3f}$")
plt.xlim(0, correction.shape[0])
plt.ylim(0.9, 1.03*correction[:, mid2].max())
plt.grid()
plt.legend(loc="upper center")
plt.savefig(results/f"flat/{label}_col.pdf")
plt.show()
plt.close()

difference = correction - g_fit
diff_max = max([abs(difference.max()), abs(difference.min())])

plt.figure(figsize=(5,5), tight_layout=True)
img = plt.imshow(difference)
plt.xticks([])
plt.yticks([])
colorbar_here = plot.colorbar(img)
colorbar_here.set_label("Correction factor (observed - fit)")
colorbar_here.update_ticks()
plt.savefig(results/f"flat/{label}_correction_difference.pdf")
plt.show()
plt.close()

plt.figure(figsize=(5,2), tight_layout=True)
plt.hist(difference.ravel(), bins=250)
plt.xlabel("Correction factor (observed - fit)")
plt.savefig(results/f"flat/{label}_correction_difference_hist.pdf")
plt.show()
plt.close()

plt.figure(figsize=(5,2), tight_layout=True)
plt.hist(difference.ravel() / correction.ravel(), bins=250)
plt.xlabel("Correction factor (observed - fit)/observed")
plt.savefig(results/f"flat/{label}_correction_difference_hist_relative.pdf")
plt.show()
plt.close()

vmins = [1, 1, -diff_max]
vmaxs = [correction.max(), correction.max(), diff_max]
clabels = ["$g$ (Observed)", "$g$ (Best fit)", "Residual"]
fig, axs = plt.subplots(ncols=3, figsize=(6,2), sharex=True, sharey=True, squeeze=True, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0})
for data, ax, vmin, vmax, clabel in zip([correction, g_fit, difference], axs, vmins, vmaxs, clabels):
    img = ax.imshow(data, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    colorbar_here = plot.colorbar(img)
    colorbar_here.set_label(clabel)
    colorbar_here.locator = plot.ticker.MaxNLocator(nbins=4)
    colorbar_here.update_ticks()
fig.savefig(results/f"flat/{label}_correction_combined.pdf")
plt.show()
plt.close()

print(f"RMS difference (gauss): {RMS(difference):.3f}")
print(f"RMS difference (gauss, relative): {100*RMS(difference/correction):.1f} %")

difference_raw = correction_raw - g_fit

print(f"RMS difference (raw): {RMS(difference_raw):.3f}")
print(f"RMS difference (raw, relative): {100*RMS(difference_raw/correction_raw):.1f} %")
