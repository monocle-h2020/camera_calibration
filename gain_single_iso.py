import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import raw, io
from phonecal.general import Rsquare

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")

products_gain, results_gain = products/"gain", results/"gain"
iso = io.split_iso(folder)
print("Loaded information")

names, means = io.load_means (folder  )
names, stds  = io.load_stds  (folder  )
colours      = io.load_colour(stacks  )
bias         = io.load_bias  (products)
print("Loaded data")

means -= bias

mean_reshaped = np.moveaxis(means, 0, 2)
stds_reshaped = np.moveaxis(stds , 0, 2)
print("Reshaped data")

variances = stds_reshaped**2
mean_RGBG, _ = raw.pull_apart(mean_reshaped, colours)
vars_RGBG, _ = raw.pull_apart(variances    , colours)
print("Split data in RGBG")

mean_mean = mean_RGBG.mean(axis=(1,2))
vars_mean = vars_RGBG.mean(axis=(1,2))
mean_stds = mean_RGBG.std (axis=(1,2))
vars_stds = vars_RGBG.std (axis=(1,2))
print("Meaned data")

mean_errors = mean_stds / np.sqrt(mean_RGBG.shape[1] * mean_RGBG.shape[2])
vars_errors = vars_stds / np.sqrt(vars_RGBG.shape[1] * vars_RGBG.shape[2])
print("Calculated errors")

if phone["software"]["bias"]:
    fit_min = 0
else:
    fit_min = 10
fit_max = 0.6 * 2**phone["camera"]["bits"]
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
print("Performed fit")

fit_measured = np.polyval(fit, mean_fit)
R2 = Rsquare(vars_fit, fit_measured)
print(f"  R^2 = {R2:.4f}")

plt.figure(figsize=(3.3,3), tight_layout=True)
x = np.logspace(-1, 4, 500)
y = np.polyval(fit, x)
yerr = np.sqrt(fiterr[0]**2 * x**2 + fiterr[1]**2)
plt.axvline(fit_min, ls="--", c="0.5")
plt.axvline(fit_max, ls="--", c="0.5")
plt.plot(x, y, c='k', zorder=2)
#plt.axhline(fit[1], c='k', ls="--", label=f"$RON = ({RON:.2f} \pm {RONerr:.2f})$ e$^-$", zorder=0)
plt.fill_between(x, y-yerr, y+yerr, color="0.5", zorder=0)
plt.errorbar([-1], [-1], xerr=[10], yerr=[10], fmt='o', c='k')
for j in range(4):
    c = "rgbg"[j]
    plt.errorbar(mean_mean[j], vars_mean[j], xerr=mean_errors[j], yerr=vars_errors[j], fmt=f"{c}o", zorder=1)
plt.xscale("log") ; plt.yscale("log")
plt.xlim(0.1, 2**phone["camera"]["bits"]) ; plt.ylim(ymin=0.9*fit[1])
plt.xlabel("Mean (ADU)")
plt.ylabel("Variance (ADU$^2$)")
plt.title(f"{phone['device']['name']}, ISO speed {iso}\n$G = ({fit[0]:.2f} \pm {fiterr[0]:.2f})$ ADU/e$^-$")
plt.savefig(results_gain/f"gain_curve_iso{iso}.pdf")
plt.close()
print("Made plot")

save_to = (products_gain/folder.stem).with_suffix(".npy")
results = np.array([gain, gainerr])
np.save(save_to, results)
print("Saved results")
