import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from ispex import raw, plot, io
from ispex.general import gaussMd

folder = argv[1]
handle = argv[1].split("/")[2]
arrs, colors = io.load_dng_many(f"{folder}/*.dng", return_colors=True)

mean = arrs.mean(axis=0).astype(np.float32)  # mean per x,y
plot.bitmap(mean, saveto=f"results/bias/Bias_mean_{handle}.png")
np.save(f"results/bias/bias_mean_{handle}.npy", mean)

stds = arrs.std(axis=0, dtype=np.float32)
plot.bitmap(stds, saveto=f"results/bias/Bias_std_im_{handle}.png")
np.save(f"results/bias/bias_stds_{handle}.npy", stds)

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
plt.hist(mean.ravel(), bins=np.linspace(518, 538, 100), color='k')
plt.xlabel(r"$\mu$")
plt.xlim(525, 531)
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
    mG = gaussMd(RGBG[...,j], sigma=10)
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
    sG = gaussMd(RGBG[...,j], sigma=10)
    plt.figure(figsize=(20,10), tight_layout=True)
    img = plt.imshow(sG, cmap=plot.cmaps[c+"r"])
    plot.colorbar(img)
    plt.savefig(f"results/bias/Bias_std_gauss_{handle}_{c}{X}.png")
    plt.close()
