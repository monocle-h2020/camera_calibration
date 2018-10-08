import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import raw, io
from phonecal.general import Rsquare

folder = argv[1]
iso = io.split_iso(folder)

names, means = io.load_means (folder)
names, stds  = io.load_stds  (folder)
colours      = io.load_colour(folder)

means -= 528  # bias correction

mean_reshaped = np.moveaxis(means, 0, 2)
stds_reshaped = np.moveaxis(stds , 0, 2)

variances = stds_reshaped**2
mean_RGBG, _ = raw.pull_apart(mean_reshaped, colours)
vars_RGBG, _ = raw.pull_apart(variances    , colours)

mean_mean = mean_RGBG.mean(axis=(1,2))
vars_mean = vars_RGBG.mean(axis=(1,2))
mean_stds = mean_RGBG.std (axis=(1,2))
vars_stds = vars_RGBG.std (axis=(1,2))

mean_errors = mean_stds / np.sqrt(mean_RGBG.shape[1] * mean_RGBG.shape[2])
vars_errors = vars_stds / np.sqrt(vars_RGBG.shape[1] * vars_RGBG.shape[2])

fit_min = 0
fit_max = 2000
ind = np.where((mean_mean > fit_min) & (mean_mean < fit_max))
mean_fit        = mean_mean  [ind].ravel()
vars_fit        = vars_mean  [ind].ravel()
vars_errors_fit = vars_errors[ind].ravel()

try:
    fit, cov = np.polyfit(mean_fit, vars_fit, 1, w=1/vars_errors_fit, cov=True)
except ValueError:
    print("NO COVARIANCE MATRIX")
    fit = np.polyfit(mean_mean[ind], vars_mean[ind], 1)
    cov = np.tile(np.nan, (2,2))
fiterr = np.sqrt(np.diag(cov))
gain = 1/fit[0]
gainerr = gain**2 * fiterr[0]
RON  = gain * np.sqrt(fit[1])
RONerr = np.sqrt(gainerr**2 * fit[1] + 0.25 * fiterr[1]**2 * gain**2 / fit[1])

fit_measured = np.polyval(fit, mean_fit)
R2 = Rsquare(vars_fit, fit_measured)

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
    plt.errorbar(mean_mean[j], vars_mean[j], xerr=mean_errors[j], yerr=vars_errors[j], fmt=f"{c}o")
plt.xscale("log") ; plt.yscale("log")
plt.xlim(0.1, 4096) ; plt.ylim(ymin=0.9*fit[1])
plt.xlabel("Mean (ADU)")
plt.ylabel("Variance (ADU$^2$)")
plt.title(f"$R^2 = {R2:.4f}$")
plt.legend(loc="upper left")
plt.savefig(f"results/gain_new/G_RON_iso{iso}.png")
plt.show()

save_to = folder.replace("stacks", "products").strip("/")
results = np.array([gain, gainerr])
np.save(f"{save_to}.npy", results)
