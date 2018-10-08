import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import raw, plot, io
from phonecal.general import gaussMd

folder = argv[1]
isos, means = io.load_means (folder, retrieve_value=io.split_iso, file=True)
isos, stds  = io.load_stds  (folder, retrieve_value=io.split_iso, file=True)
colours     = io.load_colour(folder)

low_iso = isos.argmin()
high_iso= isos.argmax()

saveto = folder.replace("stacks", "products").strip("/")
np.save(f"{saveto}.npy", means[low_iso])

raise Exception

plot.bitmap(mean, saveto=f"results/bias/Bias_mean_{handle}.png")
plot.bitmap(stds, saveto=f"results/bias/Bias_std_{handle}.png")

plt.figure(figsize=(10,7), tight_layout=True)
plt.hist(stds.ravel(), bins=np.arange(0, 25, 0.2), color='k')
plt.xlabel(r"$\sigma$")
plt.xlim(0,25)
plt.yscale("log")
plt.ylabel("Frequency")
plt.ylim(ymin=0.9)
plt.grid(ls="--")
plt.savefig(f"results/bias/Bias_std_hist_{handle}.png")
plt.close()

plt.figure(figsize=(10,7), tight_layout=True)
plt.hist(mean.ravel(), bins=np.linspace(513, 543, 100), color='k')
plt.xlabel(r"$\mu$")
plt.xlim(513, 543)
plt.yscale("log")
plt.ylabel("Frequency")
plt.ylim(ymin=0.9)
plt.grid(ls="--")
plt.savefig(f"results/bias/Bias_mean_hist_{handle}.png")
plt.close()

mG = gaussMd(mean, sigma=10)
plt.figure(figsize=(20,10), tight_layout=True)
img = plt.imshow(mG)
plot.colorbar(img)
plt.savefig(f"results/bias/Bias_mean_gauss_{handle}.png")
plt.close()

RGBG, _ = raw.pull_apart(mean, colors)
for j,c in enumerate("RGBG"):
    X = "2" if j == 3 else ""
    mG = gaussMd(RGBG[j], sigma=10)
    plt.figure(figsize=(20,10), tight_layout=True)
    img = plt.imshow(mG, cmap=plot.cmaps[c+"r"])
    plot.colorbar(img)
    plt.savefig(f"results/bias/Bias_mean_gauss_{handle}_{c}{X}.png")
    plt.close()

sG = gaussMd(stds, sigma=10)
plt.figure(figsize=(20,10), tight_layout=True)
img = plt.imshow(sG)
plot.colorbar(img)
plt.savefig(f"results/bias/Bias_std_gauss_{handle}.png")
plt.close()

RGBG, _ = raw.pull_apart(stds, colors)
for j,c in enumerate("RGBG"):
    X = "2" if j == 3 else ""
    sG = gaussMd(RGBG[j], sigma=10)
    plt.figure(figsize=(20,10), tight_layout=True)
    img = plt.imshow(sG, cmap=plot.cmaps[c+"r"])
    plot.colorbar(img)
    plt.savefig(f"results/bias/Bias_std_gauss_{handle}_{c}{X}.png")
    plt.close()
