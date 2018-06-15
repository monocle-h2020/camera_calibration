import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from ispex import raw, plot, io
from ispex.general import bin_centers
from glob import glob
from scipy.stats import binned_statistic

folder = argv[1]
arrs, colors = io.load_dng_many(f"{folder}/*.dng", return_colors=True)

bias, ron = io.load_bias_ron(iso=argv[2])

mean = arrs.mean(axis=0).astype(np.float32) - bias  # mean per x,y
plot.bitmap(mean, saveto="Gain_mean.png")

stds = arrs.std(axis=0, dtype=np.float32)
plot.bitmap(stds, saveto="Gain_std_im.png")

var = stds**2
RGBG_var, _ = raw.pull_apart(var, colors)
RGBG_mean, _ = raw.pull_apart(mean, colors)

fig, axs = plt.subplots(2,2, sharex=True, sharey=True, figsize=(27,16), tight_layout=True)
for j in range(4):
    ax = axs.ravel()[j]
    div = RGBG_var[...,j].ravel() / (RGBG_mean[...,j].ravel())
    div = div[~np.isinf(div)]
    ax.hist(div, bins=np.arange(0,8,0.1), color="RGBG"[j], density=True)
    ax.set_xlim(0, 8)
#    ax.set_ylim(0,1)
    ax.grid()
axs[1,0].set_xlabel(r"$\sigma^2 / (\mu-B)$")
axs[1,1].set_xlabel(r"$\sigma^2 / (\mu-B)$")
axs[0,0].set_ylabel("$f$")
axs[1,0].set_ylabel("$f$")
fig.savefig("Gain_hist.png")
plt.close()

var -= ron**2
RGBG_var, _ = raw.pull_apart(var, colors)
fig, axs = plt.subplots(2,2, sharex=True, sharey=True, figsize=(27,16), tight_layout=True)
for j in range(4):
    ax = axs.ravel()[j]
    div = RGBG_var[...,j].ravel() / (RGBG_mean[...,j].ravel())
    div = div[~np.isinf(div)]
    ax.hist(div, bins=np.arange(0,8,0.1), color="RGBG"[j], density=True)
    ax.set_xlim(0, 8)
#    ax.set_ylim(0,1)
    ax.grid()
axs[1,0].set_xlabel(r"$(\sigma^2 - \sigma_R^2) / (\mu-B)$")
axs[1,1].set_xlabel(r"$(\sigma^2 - \sigma_R^2) / (\mu-B)$")
axs[0,0].set_ylabel("$f$")
axs[1,0].set_ylabel("$f$")
fig.savefig("Gain_hist_ron.png")
plt.close()

C_range = np.arange(0, 4096, 1)
fig, axs = plt.subplots(2,2, sharex=True, sharey=True, figsize=(27,16), tight_layout=True)
for j in range(4):
    mean_per_I, bin_edges, bin_number = binned_statistic(RGBG_mean[...,j].ravel(), RGBG_var[...,j].ravel()/RGBG_mean[...,j].ravel(), statistic="mean", bins=C_range)
    std_per_I = binned_statistic(RGBG_mean[...,j].ravel(), RGBG_var[...,j].ravel()/RGBG_mean[...,j].ravel(), statistic="std", bins=C_range).statistic
    nr_per_I = binned_statistic(RGBG_mean[...,j].ravel(), RGBG_var[...,j].ravel()/RGBG_mean[...,j].ravel(), statistic="count", bins=C_range).statistic

    bc = bin_centers(bin_edges)
    idx = np.where(nr_per_I > 100)

    ax = axs.ravel()[j]
    ax.errorbar(bc[idx], mean_per_I[idx], yerr=std_per_I[idx]/np.sqrt(nr_per_I[idx]-1), color="RGBG"[j], fmt="o")
    ax.grid()
axs[1,0].set_xlabel(r"$\mu_C - B$")
axs[1,1].set_xlabel(r"$\mu_C - B$")
axs[0,0].set_ylabel(r"$(\sigma_C^2 - \sigma_B^2) / (\mu_C - B)$")
axs[1,0].set_ylabel(r"$(\sigma_C^2 - \sigma_B^2) / (\mu_C - B)$")
axs[0,0].set_ylim(0, 8)
#axs[0,0].set_yticks(np.arange(0,1.1, 0.1))
axs[0,0].set_xlim(0, 4096)
fig.savefig("Gain_scatter.png")
plt.close()