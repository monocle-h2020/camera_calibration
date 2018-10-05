import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from ispex import raw, plot, io
from ispex.general import bin_centers, weighted_mean
from glob import glob
from scipy.stats import binned_statistic

mainfolder = argv[1]
subfolders = glob(f"{mainfolder}/*")
iso = int(mainfolder.split("iso")[1].strip("/"))

C_range = np.arange(0, 4096, 1)

meanRGB = [[], [], []]
varRGB  = [[], [], []]
errRGB  = [[], [], []]

for folder in subfolders:
    print(folder)
    arrs, colors = io.load_dng_many(f"{folder}/*.dng", return_colors=True)
    mean = arrs.mean(axis=0) # mean per x,y
    var  = arrs.var(axis=0)
    RGBG_var, _ = raw.pull_apart(var, colors)
    RGBG_mean, _ = raw.pull_apart(mean, colors)
    for j in range(4):
        colour = ["R", "G", "B", "G2"][j]
        mean = RGBG_mean[..., j].ravel()
        var  = RGBG_var[...,  j].ravel()
        mean_per_I, bin_edges, bin_number = binned_statistic(mean, var, statistic="mean", bins=C_range)
        std_per_I = binned_statistic(mean, var, statistic=np.std, bins=C_range).statistic
        std_per_I[std_per_I == 0] = np.inf
        nr_per_I = binned_statistic(mean, var, statistic="count", bins=C_range).statistic
        bc = bin_centers(bin_edges)
        idx = np.where((nr_per_I > 500) & (bc > 550) & (bc < 3750))# & (bc >= fitmin) & (bc < fitmax))
        if len(idx[0]) < 100:
            continue

        p,cov = np.polyfit(bc[idx], mean_per_I[idx], 1, w=np.sqrt(nr_per_I[idx]-1)/std_per_I[idx], cov=True)
        fitted = np.polyval(p, bc)
        errs = (mean_per_I[idx]/fitted[idx] - 1)**2
        rel_rms = np.sqrt(np.nanmean(errs))
        err = rel_rms * p[0]

        idx2 = np.where((mean > 550) & (mean < 3750))
        fitted2 = np.polyval(p, mean[idx2])
        var2 = var[idx2]
        err2 = np.median(np.abs(var2/fitted2 - 1)) * p[0]
        print(p, np.sqrt(np.diag(cov)), err, err2)

        err3 = np.median(np.abs(mean_per_I[idx] / fitted[idx] - 1)) * p[0]

        plt.figure(figsize=(10,6), tight_layout=True)
        plt.hexbin(mean, var, mincnt=1, extent=(0, 4096, 0, 4096*4), cmap=plot.cmaps["RGBG"[j] + "r"])
        plt.errorbar(bc, mean_per_I,  yerr=std_per_I/np.sqrt(nr_per_I), fmt="o", color="k")
        plt.plot(bc, fitted, ls="--", c="k", lw=2, zorder=3)
        plt.xlim(0, 4096)
        plt.ylim(0, 4096*4)
        plt.xlabel(r"$\mu$")
        plt.ylabel(r"$\sigma^2$")
        plt.title(f"$G = {p[0]:.2f} \pm {err3:.2f}$")
        plt.savefig(f"results/gain/iso{iso}_{folder.split('pol')[-1]}_{colour}.png")
        plt.close()

        if j == 3: j = 1
        meanRGB[j].append(p[0])
        varRGB[j].append(np.sqrt(cov[0,0]))
        errRGB[j].append(err3)

meanRGB = [np.array(x) for x in meanRGB]
varRGB  = [np.array(x) for x in varRGB ]
errRGB  = [np.array(x) for x in errRGB ]
gains = np.zeros(3)
gainerrs = np.zeros(3)
for j in range(3):
    gains[j], gainerrs[j] = weighted_mean(meanRGB[j], 1/errRGB[j]**2)
    print(f"{gains[j]:.3f} +- {gainerrs[j]:.3f}")

np.save(f"results/gain/values/iso{iso}.npy", np.stack((gains, gainerrs)))

rgb = "RGB"
for p0, p1 in [(0,1), (0,2), (1,2)]:
    c0, c1 = rgb[p0], rgb[p1]
    diff = np.abs(gains[p0] - gains[p1])
    differr = np.sqrt(gainerrs[p0]**2 + gainerrs[p1]**2)
    print(f"{c0} - {c1}: {diff:.3f} +- {differr:.3f} ({diff/differr:.1f} sigma)")