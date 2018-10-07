import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import raw, plot, io
from phonecal.general import bin_centers, weighted_mean, Rsquare
from glob import glob
from scipy.stats import binned_statistic

folder_main = argv[1]
try:
    iso = int(folder_main.split("iso")[-1].strip("/"))
except:
    iso = "_".join(folder_main.replace("\\", "/").split("/")).strip("_")
folders_pol = glob(folder_main+"/*")
folders_pol = [f for f in folders_pol if "." not in f]
Vs = np.zeros_like(folders_pol, dtype=np.float32)
Verrs = Vs.copy()
Ms = Vs.copy()
Merrs = Ms.copy()
maxes = Ms.copy()
VsRGBG = np.zeros((len(folders_pol), 4))
VerrsRGBG = VsRGBG.copy()
MsRGBG = VsRGBG.copy()
MerrsRGBG = MsRGBG.copy()
for i,folder in enumerate(folders_pol):
    files = glob(folder+"/*.dng")

    arrs, colors = io.load_dng_many(f"{folder}/*.dng", return_colors=True)
    arrs_RGBG = np.moveaxis(arrs, 0, 2)
    arrs_RGBG, _ = raw.pull_apart(arrs_RGBG, colors)
    arrs_RGBG = np.moveaxis(arrs_RGBG, 3, 1)

    max_here = arrs.max()
    mean = arrs.mean(axis=0).astype(np.float32)  # mean per x,y
    mean_RGBG, _ = raw.pull_apart(mean, colors)

    means = arrs.mean(axis=(1,2)) - 528
    means_RGBG = arrs_RGBG.mean(axis=(2,3)) - 528
    mean_all = means.mean()
    mean_all_RGBG = means_RGBG.mean(axis=1)
    mean_err = means.std() / np.sqrt(len(means) - 1)
    mean_err_RGBG = means_RGBG.std(axis=1) / np.sqrt(len(means) - 1)

    new_arrs = arrs - mean
    new_arrs_RGBG = arrs_RGBG - mean_RGBG[:,np.newaxis,...]

    stds = new_arrs.std(axis=(1,2))
    stds_RGBG = new_arrs_RGBG.std(axis=(2,3))
    std_mean = stds.mean()
    std_mean_RGBG = stds_RGBG.mean(axis=1)
    var = std_mean**2
    var_RGBG = std_mean_RGBG**2
    var_err = stds.std() / np.sqrt(len(stds) - 1) * 2 * std_mean
    var_err_RGBG = stds_RGBG.std(axis=1) / np.sqrt(stds_RGBG.shape[1] - 1) * 2 * std_mean_RGBG
    Vs[i], Verrs[i], Ms[i], Merrs[i], maxes[i] = var, var_err, mean_all, mean_err, max_here
    VsRGBG[i], VerrsRGBG[i], MsRGBG[i], MerrsRGBG[i] = var_RGBG, var_err_RGBG, mean_all_RGBG, mean_err_RGBG

    print(f"{(i+1)/len(folders_pol)*100:.0f}%", end=" ")

print("")

Ms = MsRGBG.ravel()
Vs = VsRGBG.ravel()
Verrs = VerrsRGBG.ravel()

fit_min = 0
fit_max = 2000
ind = np.where((Ms > fit_min) & (Ms < fit_max))
try:
    fit, cov = np.polyfit(Ms[ind], Vs[ind], 1, w=1/Verrs[ind], cov=True)
except ValueError:
    print("NO COVARIANCE MATRIX")
    fit = np.polyfit(Ms[ind], Vs[ind], 1)
    cov = np.tile(np.nan, (2,2))
fiterr = np.sqrt(np.diag(cov))
gain = 1/fit[0]
gainerr = gain**2 * fiterr[0]
RON  = gain * np.sqrt(fit[1])
RONerr = np.sqrt(gainerr**2 * fit[1] + 0.25 * fiterr[1]**2 * gain**2 / fit[1])

fit_measured = np.polyval(fit, Ms)
R2 = Rsquare(Vs[ind], fit_measured[ind])

plt.figure(figsize=(7,5), tight_layout=True)
x = np.logspace(-1, 4, 500)
y = np.polyval(fit, x)
yerr = np.sqrt(fiterr[0]**2 * x**2 + fiterr[1]**2)
plt.axvline(fit_min, ls="--", c="0.5")
plt.axvline(fit_max, ls="--", c="0.5")
plt.plot(x, y, c='k', label=f"$G = {gain:.3f}$ e$^-$/ADU")
plt.axhline(fit[1], c='k', ls="--", label=f"$RON = {RON:.3f}$ e$^-$")
plt.fill_between(x, y-yerr, y+yerr, color="0.5", label=f"$\sigma_G = {gainerr:.3f}$\n$\sigma_R = {RONerr:.2}$")
plt.errorbar([-1], [-1], xerr=[10], yerr=[10], fmt='o', c='k', label=f"$ISO = {iso}$")
for j in range(4):
    c = "rgbg"[j]
    plt.errorbar(MsRGBG[:,j], VsRGBG[:,j], xerr=MerrsRGBG[:,j], yerr=VerrsRGBG[:,j], fmt=f"{c}o")
plt.xscale("log") ; plt.yscale("log")
plt.xlim(0.1, 4096) ; plt.ylim(ymin=0.9*fit[1])
plt.xlabel("Mean (ADU)")
plt.ylabel("Variance (ADU$^2$)")
plt.title(f"$R^2 = {R2:.4f}$")
plt.legend(loc="upper left")
plt.savefig(f"results/gain_new/G_RON_iso{iso}.png")
plt.show()

results = np.array([gain, gainerr, RON, RONerr])
np.save(f"{folder_main}/gain_ron.npy", results)
