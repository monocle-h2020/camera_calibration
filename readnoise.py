import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from ispex.general import cut
from ispex.gamma import polariser_angle, I_range, cos4f, malus, find_I0, pixel_angle
from ispex import raw, plot, io
from scipy.optimize import curve_fit
from scipy.stats import binned_statistic
from glob import glob

folder = argv[1]
files = glob(folder+"/*.dng")
imgs = [io.load_dng_raw(filename) for filename in files]
colors = imgs[0].raw_colors
RGBGs = np.empty((len(files), colors.shape[0]//2, colors.shape[1]//2, 4), dtype=np.int16)
for j, img in enumerate(imgs):
    RGBGs[j] = raw.pull_apart(img.raw_image, colors)[0]
del imgs

mean = RGBGs.mean(axis=0).astype(np.float32)  # mean per x,y,colour

RGBG_means = mean.mean(axis=(0,1))

fig, axs = plt.subplots(2,2, sharex=True, sharey=True, tight_layout=True, figsize=(10,7))
for j in range(4):
    ax = axs.ravel()[j]
    ax.hist(mean[...,j].ravel(), bins=np.linspace(528-5,528+5,50), color="RGBG"[j])
    mu, sigma = np.mean(mean[...,j]), np.std(mean[...,j])
    ax.grid(ls="--")
    ax.set_yscale("log")
    ax.set_xlim(528-5, 528+5)
axs[0,0].set_title(f"$\\mu = {mu:.2f}$")
axs[0,1].set_title(f"$\\sigma = {sigma:.2f}$")
plt.savefig("Bias_hist_RGBG.png")
plt.close("all")

plt.figure(figsize=(10,7), tight_layout=True)
plt.hist(mean.ravel(), bins=np.linspace(528-5,528+5,50), color='k')
mu, sigma = np.mean(mean), np.std(mean)
plt.grid(ls="--")
plt.yscale("log")
plt.xlim(528-5, 528+5)
plt.title(f"$\\mu = {mu:.2f}$, $\\sigma = {sigma:.2f}$")
plt.savefig("Bias_hist_total.png")
plt.close("all")

label = ["R", "G", "B", "G2"]
for j in range(4):
    print("---")
    print(label[j])
    tran_amp_avg = np.zeros_like(RGBGs[0,...,j], dtype=np.float32)
    for i in range(len(files)):
        print(i)
        norm = RGBGs[i,...,j] - mean[...,j]

        plt.figure(figsize=(12, 7), tight_layout=True)
        nr, bins = plt.hist(norm.ravel(), bins=np.linspace(-15,16,50), density=True, color="RGBG"[j])[:2]
        mu, sigma = norm.mean(), norm.std()
        plt.yscale("log")
        plt.xlim(-15, 15)
        plt.xlabel("Bias - Mean bias")
        plt.ylabel("Frequency")
        plt.title(f"Read-out noise; $\\mu = {mu:.1f}$ ; $\\sigma = {sigma:.1f}$")
        plt.savefig(f"Readnoise_{label[j]}_{i}.png")
        plt.close()

        plt.figure(figsize=(20,16), dpi=96, tight_layout=True)
        plt.imshow(norm)
        plt.axis("off")
        plt.savefig(f"Readnoise_{label[j]}_img_{i}.png", dpi=96, transparent=True)
        plt.close()

        tran = np.fft.fftshift(np.fft.fft2(norm))
        amp = np.abs(tran)
        plt.figure(figsize=(12,10), tight_layout=True)
        plt.imshow(amp)
        plt.title(f"FFT of read-out noise in {label[j]}")
        plt.savefig(f"Readnoise_{label[j]}_fft_{i}.png")
        plt.close()

        tran[amp < np.percentile(amp, 99)] = 0
        b = np.fft.ifft2(np.fft.ifftshift(tran))
        plt.figure(figsize=(12,10), tight_layout=True)
        plt.imshow(b.real)
        plt.title(f"Inverse FFT of read-out noise in {label[j]}")
        plt.savefig(f"Readnoise_{label[j]}_ifft_{i}.png")
        plt.close()

        tran_amp_avg += amp
    tran_amp_avg /= i
    plt.figure(figsize=(12,10), tight_layout=True)
    plt.imshow(tran_amp_avg)
    plt.title(f"Mean FFT of read-out noise in {label[j]}")
    plt.savefig(f"Readnoise_{label[j]}_fft_mean.png")
    plt.close()