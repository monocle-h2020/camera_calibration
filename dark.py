import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import raw, plot, io, iso
from phonecal.general import Rsquare, gaussMd

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")
ISO = io.split_iso(folder)

times, means= io.load_means(folder, retrieve_value=io.split_exposure_time)
print("Loaded means")
colours     = io.load_colour(stacks)
print(f"Loaded data: {len(times)} exposure times")

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
plt.ylabel("Mean (ADU)")
plt.xlim(0, time_max)
plt.ticklabel_format(useOffset=False)
plt.title(f"Ensemble fit : $R^2 = {R2:.2f}$")
plt.savefig(results/f"dark/ensemble_iso{ISO}.pdf")
plt.show()
plt.close()
print("Saved ensemble scatter plot")

raise Exception

mean_reshaped = means.reshape((means.shape[0], -1))  # as list
fit_separate = np.polyfit(times, mean_reshaped, 1)  # linear fit to every pixel
print("Fitted data")
dark_separate, bias_separate = fit_separate
dark_reshaped = dark_separate.reshape(means[0].shape)

dark_gauss = gaussMd(dark_reshaped, 10)
plot.show_image(dark_gauss, colorbar_label="Dark current (ADU/s)", saveto=results/f"dark/map_iso{ISO}.pdf")
print("Saved Gauss map")

dark_RGBG, _= raw.pull_apart(dark_reshaped, colours)
plot.hist_bias_ron_kRGB(dark_RGBG, xlim=(-25, 50), xlabel="Dark current (ADU/s)", saveto=results/f"dark/histogram_RGB_iso{ISO}.pdf")
del dark_RGBG
print("Saved RGB histogram")

plt.figure(figsize=(3.3, 3), tight_layout=True)
plt.hist(dark_separate, bins=250, color='k', edgecolor='k')
plt.yscale("log")
plt.xlabel("Dark current (ADU/s)")
plt.ylabel("Frequency")
plt.savefig(results/f"dark/histogram_iso{ISO}.pdf")
plt.close()
print("Saved ADU histogram")

raise Exception

gain_table = io.read_gain_lookup_table(results)
G, G_err = gain.get_gain(gain_table, iso)

dark_electrons = dark_separate * G
plt.figure(figsize=(3.3, 3), tight_layout=True)
plt.hist(dark_electrons, bins=250, color='k', edgecolor='k')
plt.yscale("log")
plt.xlabel("Dark current (e$^-$/s)")
plt.ylabel("Frequency")
plt.savefig(results/f"dark/electrons_histogram_iso{iso}.pdf")
plt.close()
print("Saved e- histogram")

print(f"ISO {iso} ; mean {dark_separate.mean():.3f} ADU/s == {dark_electrons.mean():.3f} e-/s ; std {dark_separate.std():.3f} ADU/s == {dark_electrons.std():.3f} e-/s ; RMS {np.sqrt(np.mean(dark_separate**2)):.3f} ADU/s == {np.sqrt(np.mean(dark_electrons**2)):.3f} e-/s")
