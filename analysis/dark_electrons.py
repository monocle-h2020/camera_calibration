import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import raw, plot, io, iso
from spectacle.general import Rsquare, gaussMd, gauss_nan

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.load_metadata(root)
ISO = io.split_iso(folder)

times, means= io.load_means(folder, retrieve_value=io.split_exposure_time)
print("Loaded means")
colours     = io.load_colour(stacks)
print(f"Loaded data: {len(times)} exposure times")

ISO_gain, gain = io.read_gain_table(results/"gain"/"table_iso50.npy")  # hard-coded for now
iso_lookup_table = io.read_iso_lookup_table(products)
gain_normalised = iso.normalise_single_iso(gain, ISO, iso_lookup_table)
gain_gauss = gauss_nan(gain_normalised, sigma=(0,5,5))
gain_image = raw.put_together_from_colours(gain_gauss, colours)

means = means / gain_image

mean_mean = means.mean(axis=(1,2))
mean_std  = means.std (axis=(1,2))
mean_err  = mean_std / np.sqrt(means[0].size - 1)

fit_ensemble, cov_ensemble = np.polyfit(times, mean_mean, 1, w=1/mean_err, cov=True)

time_max = 1/phone["software"]["1/t max"]
time_range = np.linspace(0, 1.05*time_max, 10)
time_range_fit = np.polyval(fit_ensemble, time_range)
time_range_fit_err = np.sqrt(time_range**2 * cov_ensemble[0,0] + cov_ensemble[1,1])
times_fit = np.polyval(fit_ensemble, times)
R2 = Rsquare(mean_mean, times_fit)
print(f"Fit ensemble: R^2 = {R2:.2f}")

plt.figure(figsize=(3.3, 3), tight_layout=True)
plt.errorbar(times, mean_mean, yerr=mean_err, fmt="ko")
plt.plot(time_range, time_range_fit, c="k")
plt.fill_between(time_range, time_range_fit-time_range_fit_err, time_range_fit+time_range_fit_err, color="0.5", zorder=0)
plt.xlabel("Exposure time (s)")
plt.ylabel("Mean (e-)")
plt.xlim(0, time_max)
plt.ticklabel_format(useOffset=False)
plt.title(f"Ensemble fit : $R^2 = {R2:.2f}$")
plt.savefig(results/f"dark/electrons_ensemble_iso{ISO}.pdf")
plt.show()
plt.close()
print("Saved ensemble scatter plot")

mean_reshaped = means.reshape((means.shape[0], -1))  # as list
fit_separate = np.polyfit(times, mean_reshaped, 1)  # linear fit to every pixel
print("Fitted data")
dark_separate, bias_separate = fit_separate
dark_reshaped = dark_separate.reshape(means[0].shape)

dark_gauss = gaussMd(dark_reshaped, 25)
plot.show_image(dark_gauss, colorbar_label="Dark current (e-/s)", saveto=results/f"dark/electrons_map_iso{ISO}.pdf")
print("Saved Gauss map")

dark_RGBG, _= raw.pull_apart(dark_reshaped, colours)
plot.hist_bias_ron_kRGB(dark_RGBG, xlim=(-25, 50), xlabel="Dark current (e-/s)", saveto=results/f"dark/electrons_histogram_RGB_iso{ISO}.pdf")
del dark_RGBG
print("Saved RGB histogram")

plt.figure(figsize=(3.3, 3), tight_layout=True)
plt.hist(dark_separate, bins=np.linspace(-50,50,250), color='k', edgecolor='k')
plt.yscale("log")
plt.xlabel("Dark current (e-/s)")
plt.ylabel("Frequency")
plt.savefig(results/f"dark/electrons_histogram_iso{ISO}.pdf")
plt.show()
plt.close()
print("Saved e- histogram")

print("ISO", ISO)
print(f"Mean  = {dark_separate.mean():+.3f} e-/s")
print(f"Std   = {dark_separate. std():+.3f} e-/s")
print(f"RMS   = {np.sqrt(np.mean(dark_separate**2)):+.3f} e-/s")
print(f"P_99.9= {np.percentile(np.abs(dark_separate.ravel()), 99.9):+.3f} e-/s")
