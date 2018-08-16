import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from ispex import raw, plot, io
from ispex.general import bin_centers, weighted_mean
from ispex.gamma import malus, malus_error
from glob import glob
from scipy.optimize import curve_fit
from PIL import Image

folder_main = argv[1]

subfolders = glob(folder_main+"/*")

Vs = np.zeros_like(subfolders, dtype=np.float32)
Verrs = Vs.copy()
Ms = Vs.copy()
Merrs = Ms.copy()

jVs = np.tile(0, (len(subfolders), 3)).astype(np.float32)
jVerrs = jVs.copy()
jMs = jVs.copy()
jMerrs = jVs.copy()

for i,folder in enumerate(subfolders):
    arrs, colors = io.load_dng_many(folder+"/*.dng", return_colors=True)

    mean = arrs.mean(axis=0).astype(np.float32)  # mean per x,y

    means = arrs.mean(axis=(1,2)) - 528
    mean_all = means.mean()
    mean_err = means.std() / np.sqrt(len(means) - 1)

    new_arrs = arrs - mean

    stds = new_arrs.std(axis=(1,2))
    std_mean = stds.mean()
    var = std_mean**2
    var_err = stds.std() / np.sqrt(len(stds) - 1) * 2 * std_mean

    Vs[i], Verrs[i], Ms[i], Merrs[i] = var, var_err, mean_all, mean_err

    JPGs = glob(folder+"/*.jpg")
    jpgarrs = [plt.imread(jpg) for jpg in JPGs]
    jpgarrs = np.stack(jpgarrs)
    jpgmean = jpgarrs.mean(axis=0).astype(np.float32) # mean per x,y,C

    jpgmeans = jpgarrs.mean(axis=(1,2))
    jpgmean_all = jpgmeans.mean(axis=0)
    jpgmean_err = jpgmeans.std(axis=0) / np.sqrt(len(jpgmeans) - 1)

    jpgstds = jpgarrs.std(axis=(1,2))
    jpgstd_mean = jpgstds.mean(axis=0)
    jpg_var = jpgstd_mean**2
    jpg_var_err = jpgstds.std(axis=0) / np.sqrt(len(jpgstds) - 1) * 2 * jpgstd_mean

    jVs[i], jVerrs[i], jMs[i], jMerrs[i] = jpg_var, jpg_var_err, jpgmean_all, jpgmean_err

    print(f"{(i+1)/len(subfolders)*100:.0f}%", end=" ")

fig = plt.figure(figsize=(7,5), tight_layout=True)

ax1 = fig.add_subplot(111, label="1")
for j, c in enumerate("rgb"):
    ax1.errorbar(jMs[:,j], jVs[:,j], xerr=jMerrs[:,j], yerr=jVerrs[:,j], fmt=f"{c}o-", label="JPG")
ax1.errorbar([-100, -100], [-100, -100], yerr=[0.1, 0.1], fmt="ko-", label="DNG")

ax1.legend(loc="lower right")
ax1.set_xlabel("JPEG mean", color="C0", size="large")
ax1.set_ylabel("JPEG variance", color="C0", size="large")
ax1.tick_params(axis="x", colors="C0")
ax1.tick_params(axis="y", colors="C0")
ax1.set_xlim(0, 255)
ax1.set_ylim(ymin=0)

ax2 = fig.add_subplot(111, label="2", frame_on=False)
ax2.errorbar(Ms, Vs, xerr=Merrs, yerr=Verrs, fmt="ko-", lw=2)
ax2.xaxis.tick_top()
ax2.yaxis.tick_right()
ax2.set_xlabel("DNG mean")
ax2.set_ylabel("DNG variance")
ax2.xaxis.set_label_position("top")
ax2.yaxis.set_label_position("right")
ax2.set_xlim(xmin=0)
ax2.set_ylim(ymin=0)
plt.show()