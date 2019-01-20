import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import raw, io, plot
from phonecal.general import Rsquare, gaussMd, gauss_nan

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
mean_RGBG, offsets = raw.pull_apart(mean_reshaped, colours)
vars_RGBG, offsets = raw.pull_apart(variances    , colours)
print("Split data in RGBG")

fit_max = 0.95 * 2**phone["camera"]["bits"]

gains = np.tile(np.nan, mean_RGBG.shape[:-1])
rons  = gains.copy()

for i in range(mean_RGBG.shape[0]):       
    for j in range(mean_RGBG.shape[1]):
        for k in range(mean_RGBG.shape[2]):
            m = mean_RGBG[i,j,k] ; v = vars_RGBG[i,j,k]
            ind = np.where(m < fit_max)
            try:
                gains[i,j,k], rons[i,j,k] = np.polyfit(m[ind], v[ind], 1, w=1/m[ind])
            except:
                pass
        
        if j%650:
            print(f"{100*( (i*mean_RGBG.shape[1] + j) *mean_RGBG.shape[2])/gains.size:.1f}%")

gains_gauss = gauss_nan(gains, sigma=(0,5,5))
rons_gauss  = gauss_nan(rons , sigma=(0,5,5))

plot.hist_bias_ron_kRGB(gains, xlim=(0,8), xlabel="Gain (ADU/e$^-$)", saveto=results_gain/f"hist_iso{iso}.pdf")
print("Made histogram")

vmin, vmax = np.nanmin(gains_gauss), np.nanmax(gains_gauss)
plot.show_RGBG(gains_gauss, colorbar_label=25*" "+"Gain (ADU/e$^-$)", vmin=vmin, vmax=vmax, saveto=results_gain/f"map_iso{iso}.pdf")
print("Made map")

gains_combined = raw.put_together(*gains, offsets)
gains_combined_gauss = gauss_nan(gains_combined, sigma=5)
plt.figure(figsize=(3.3,3), tight_layout=True)
im = plt.imshow(gains_combined_gauss)
plt.xticks([])
plt.yticks([])
cbar = plot.colorbar(im)
cbar.set_label("Gain (ADU/e-)")
plt.savefig(results_gain/f"map_iso{iso}_combined.pdf")
plt.close()

i, j, k = 0, mean_RGBG.shape[1]//2, mean_RGBG.shape[2]//2
m, v = mean_RGBG[i,j,k], vars_RGBG[i,j,k]
ind = np.where(m < fit_max)
fit, cov = np.polyfit(m[ind], v[ind], 1, cov=True, w=1/m[ind])

c = ["R", "G", "B", "G$_2$"][i]
plt.figure(figsize=(3.3,3), tight_layout=True)
x = np.logspace(-1, 4, 500)
y = np.polyval(fit, x)
yerr = np.sqrt(cov[0,0] * x**2 + cov[1,1])
plt.axvline(fit_max, ls="--", c="0.5")
plt.scatter(m, v, c='k', zorder=1)
plt.plot(x, y, c='k', zorder=2)
plt.fill_between(x, y-yerr, y+yerr, color="0.5", zorder=0)
plt.xscale("log") ; plt.yscale("log")
plt.xlim(0.1, 2**phone["camera"]["bits"])
plt.ylim(ymin=0.9*fit[1])
plt.xlabel("Mean (ADU)")
plt.ylabel("Variance (ADU$^2$)")
plt.title(f"{phone['device']['name']}, {c}[{j},{k}],  ISO {iso}\n$G = ({fit[0]:.2f} \pm {np.sqrt(cov[0,0]):.2f})$ ADU/e$^-$")
plt.savefig(results_gain/f"single_curve_iso{iso}.pdf")
plt.close()
print("Made single curve plot")

v_fit = np.polyval(fit, m)
R2 = Rsquare(v, v_fit)
print(f"R^2 = {R2:.2f}")
