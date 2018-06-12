import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from ispex.general import cut, bin_centers
from ispex.gamma import polariser_angle, I_range, cos4f, malus, find_I0, pixel_angle
from ispex import raw, plot, io
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from glob import glob

folder = argv[1]
files = glob(folder+"/*.dng")
arrs = np.empty((len(files), 3024, 4032), dtype=np.uint16)
for j, file in enumerate(files):
    arrs[j] = io.load_dng_raw(file).raw_image

bias = np.load("bias_mean.npy")
ron  = np.load("bias_stds.npy")

mean = arrs.mean(axis=0).astype(np.float32) - bias  # mean per x,y
plt.figure(figsize=(mean.shape[1]/96,mean.shape[0]/96), dpi=96, tight_layout=True)
plt.imshow(mean, interpolation="none")
plt.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig("Gain_mean.png", dpi=96, transparent=True)
plt.close()

stds = np.array([np.std(arrs[:,j], axis=0).astype(np.float32) for j in range(arrs.shape[1])], dtype=np.float32)
del arrs
plt.figure(figsize=(stds.shape[1]/96,stds.shape[0]/96), dpi=96, tight_layout=True)
plt.imshow(stds, interpolation="none", aspect="equal")
plt.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig("Gain_std_im.png", dpi=96, transparent=True)
plt.close()

var = stds**2
RGBG_var, _ = raw.pull_apart(var, io.load_dng_raw(file).raw_colors)
RGBG_mean, _ = raw.pull_apart(mean, io.load_dng_raw(file).raw_colors)

fig, axs = plt.subplots(2,2, sharex=True, sharey=True, figsize=(27,16), tight_layout=True)
for j in range(4):
    ax = axs.ravel()[j]
    ax.hist( RGBG_var[...,j].ravel() / (RGBG_mean[...,j].ravel()) , bins=np.linspace(0,2,100), color="RGBG"[j], density=True)
    ax.set_xlim(0, 2)
#    ax.set_ylim(0,1)
    ax.grid()
axs[1,0].set_xlabel(r"$\sigma^2 / (\mu-B)$")
axs[1,1].set_xlabel(r"$\sigma^2 / (\mu-B)$")
axs[0,0].set_ylabel("$f$")
axs[1,0].set_ylabel("$f$")
fig.savefig("Gain_hist.png")
plt.close()

var -= ron**2
RGBG_var, _ = raw.pull_apart(var, io.load_dng_raw(file).raw_colors)
fig, axs = plt.subplots(2,2, sharex=True, sharey=True, figsize=(27,16), tight_layout=True)
for j in range(4):
    ax = axs.ravel()[j]
    ax.hist( RGBG_var[...,j].ravel() / (RGBG_mean[...,j].ravel()) , bins=np.linspace(0,2,100), color="RGBG"[j], density=True)
    ax.set_xlim(0, 2)
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
axs[0,0].set_ylabel("r$(\sigma_C^2 - \sigma_B^2) / (\mu_C - B)$")
axs[1,0].set_ylabel("r$(\sigma_C^2 - \sigma_B^2) / (\mu_C - B)$")
axs[0,0].set_ylim(0, 1)
axs[0,0].set_yticks(np.arange(0,1.1, 0.1))
axs[0,0].set_xlim(0, 1000)
fig.savefig("Gain_scatter.png")
plt.close()