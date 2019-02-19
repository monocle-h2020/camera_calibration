import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import raw, io, plot
from phonecal.general import gaussMd

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

plt.figure(figsize=(3,2), tight_layout=True)
img = plt.imshow(mRGBG[0], cmap=plot.cmaps["Rr"])
plt.xticks([])
plt.yticks([])
colorbar_here = plot.colorbar(img)
colorbar_here.set_label("Relative sensitivity")
colorbar_here.locator = plot.ticker.MaxNLocator(nbins=4)
colorbar_here.update_ticks()
plt.savefig(results/f"flat/iso{iso}_R.pdf")
plt.close()
print("Made single plot")

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
plt.savefig(results/f"flat/iso{iso}_vertical")
plt.show()
plt.close()

plt.plot(x, mRGBG[0,:,mid2], c='k')
plt.fill_between(x, mRGBG[0,:,mid2]-sRGBG[0,:,mid2], mRGBG[0,:,mid2]+sRGBG[0,:,mid2], color="0.5")
plt.xlabel("X position")
plt.ylabel("Relative sensitivity")
plt.xlim(0, mean.shape[0])
plt.ylim(0, 1.1)
plt.grid()
plt.savefig(results/f"flat/iso{iso}_horizontal")
plt.show()
plt.close()

print("Minima:", mRGBG.min(axis=(1,2)))
