import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from ispex.general import cut
from ispex.gamma import polariser_angle, I_range, cos4f, malus, find_I0, pixel_angle
from ispex import raw, plot, io
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from glob import glob

folder = argv[1]
handle = argv[1].split("/")[2]
arrs, colors = io.load_dng_many(f"{folder}/*.dng", return_colors=True)

mean = arrs.mean(axis=0).astype(np.float32)  # mean per x,y
plt.figure(figsize=(mean.shape[1]/96,mean.shape[0]/96), dpi=96, tight_layout=True)
plt.imshow(mean, interpolation="none")
plt.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(f"results/bias/Bias_mean_{handle}.png", dpi=96, transparent=True)
plt.close()
np.save(f"results/bias/bias_mean_{handle}.npy", mean)

stds = arrs.std(axis=0, dtype=np.float32)
plt.figure(figsize=(stds.shape[1]/96,stds.shape[0]/96), dpi=96, tight_layout=True)
plt.imshow(stds, interpolation="none", aspect="equal")
plt.axis("off")
plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
plt.savefig(f"results/bias/Bias_std_im_{handle}.png", dpi=96, transparent=True)
plt.close()
np.save(f"results/bias/bias_stds_{handle}.npy", stds)

plt.figure(figsize=(10,7), tight_layout=True)
plt.hist(stds.ravel(), bins=np.arange(0, 25, 0.2), density=True, color='k')
plt.xlabel(r"$\sigma$")
plt.xlim(0,25)
plt.yscale("log")
plt.ylabel("Frequency")
plt.ylim(ymin=1e-5)
plt.grid(ls="--")
plt.savefig(f"results/bias/Bias_std_hist_{handle}.png")
plt.close()

raise Exception

RGBG_std, _ = raw.pull_apart(stds, colors)
RGBG_mean, _ = raw.pull_apart(mean, colors)
for j in range(4):
    c = RGBG_std[...,j]
    label = ["R", "G", "B", "G2"][j]
    plt.figure(figsize=(c.shape[1]/96,c.shape[0]/96), dpi=96, tight_layout=True)
    plt.imshow(c, interpolation="none", aspect="equal")
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f"Bias_std_im_{label}.png", dpi=96, transparent=True)
    plt.close()

    plt.figure(figsize=(10,7), tight_layout=True)
    plt.hist(c.ravel(), bins=np.linspace(0, 8, 50), density=True, color='rgbg'[j])
    plt.xlabel(r"$\sigma_{" + label + "}$")
    plt.xlim(xmin=0)
    plt.yscale("log")
    plt.ylabel("Frequency")
    plt.grid(ls="--")
    plt.savefig(f"Bias_std_hist_{label}.png")
    plt.close()

    f = np.fft.fftshift(np.fft.fft2(c))
    plt.figure(figsize=(f.shape[1]/96,f.shape[0]/96), dpi=96, tight_layout=True)
    plt.imshow(np.abs(f), interpolation="none", aspect="equal", vmax=5e3)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f"Bias_std_fft_{label}.png", dpi=96, transparent=True)
    plt.close()

    del c
    m = RGBG_mean[...,j]
    f = np.fft.fftshift(np.fft.fft2(m))
    plt.figure(figsize=(f.shape[1]/96,f.shape[0]/96), dpi=96, tight_layout=True)
    plt.imshow(np.abs(f), interpolation="none", aspect="equal", vmax=5e3)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(f"Bias_mean_fft_{label}.png", dpi=96, transparent=True)
    plt.close()

raise Exception
plt.figure(figsize=(10,7), tight_layout=True)
plt.hist(arrs.ravel(), bins=np.arange(528.5-11,528.5+11,1), color='k')
plt.grid(ls="--")
plt.yscale("log")
plt.xticks(np.arange(528-10, 528+11, 1))
plt.xlim(528-10, 528+10)
plt.savefig("Bias_hist_total_total.png")
plt.close("all")

plt.figure(figsize=(10,7), tight_layout=True)
plt.hist(mean.ravel(), bins=np.linspace(528-10,528+10,50), color='k')
mu, sigma = np.mean(mean), np.std(mean)
plt.grid(ls="--")
plt.yscale("log")
plt.xlim(528-5, 528+5)
plt.title(f"$\\mu = {mu:.2f}$, $\\sigma = {sigma:.2f}$")
plt.savefig("Bias_hist_of_mean.png")
plt.close("all")
plt.close()

label = ["R", "G", "B", "G2"]
tran_amp_avg = np.zeros_like(mean, dtype=np.float32)
for i, a in enumerate(arrs):
    print(i)
    norm = a - mean

    plt.figure(figsize=(12, 7), tight_layout=True)
    mu, sigma = norm.mean(), norm.std()
    plt.hist(norm.ravel(), bins=np.arange(-15,16,0.5))
    plt.yscale("log")
    plt.xlim(-15, 15)
    plt.xlabel("Bias - Mean bias")
    plt.ylabel("Frequency")
    plt.title(f"Read-out noise; $\\mu = {mu:.1f}$ ; $\\sigma = {sigma:.1f}$")
    plt.savefig(f"Readnoise_{i}.png")
    plt.close()

    plt.figure(figsize=(20,16), dpi=96, tight_layout=True)
    plt.imshow(norm)
    plt.axis("off")
    plt.savefig(f"Readnoise_img_{i}.png", dpi=96, transparent=True)
    plt.close()
#
#    tran = np.fft.fftshift(np.fft.fft2(norm))
#    amp = np.abs(tran)
#    plt.figure(figsize=(20,18), tight_layout=True)
#    plt.imshow(amp)
#    plt.title(f"FFT of read-out noise")
#    plt.savefig(f"Readnoise_fft_{i}.png", dpi=96)
#    plt.close()
#
#    tran[amp < np.percentile(amp, 99)] = 0
#    b = np.fft.ifft2(np.fft.ifftshift(tran))
#    plt.figure(figsize=(12,10), tight_layout=True)
#    plt.imshow(b.real)
#    plt.title(f"Inverse FFT of read-out noise")
#    plt.savefig(f"Readnoise_ifft_{i}.png")
#    plt.close()
#
#    tran_amp_avg += amp
#
#tran_amp_avg /= i
#plt.figure(figsize=(20,18), tight_layout=True)
#plt.imshow(tran_amp_avg)
#plt.title(f"Mean FFT of read-out nois")
#plt.savefig(f"Readnoise_fft_mean.png", dpi=96)
#plt.close()