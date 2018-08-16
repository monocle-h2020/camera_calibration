import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from ispex import raw, plot, io

file = argv[1]
img = io.load_dng_raw(file)
arr = img.raw_image
col = img.raw_colors

x = np.arange(arr.shape[1])
y = np.arange(arr.shape[0])
X, Y = np.meshgrid(x, y)

RGBG, _ = raw.pull_apart(arr, col)
X_s, _  = raw.pull_apart(X,   col)

mean = RGBG.mean(axis=0)
stds = RGBG.std(axis=0)

fig, axs = plt.subplots(nrows=5, sharex=True, figsize=(7, 7), tight_layout=True, subplot_kw={"xlim": (0, arr.shape[1])})
axs[0].imshow(arr, aspect="auto")
for j, ax in enumerate(axs[1:]):
    x = X_s[...,j][0]
    ax.plot(x, mean[...,j], c="RGBG"[j])
    ax.fill_between(x, mean[...,j] - stds[...,j], mean[...,j] + stds[...,j], color="RGBG"[j], alpha=0.3)
    ax.set_ylim(0, mean.max()*1.1)
plt.show()
plt.close()