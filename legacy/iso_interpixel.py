import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import raw, io, plot
from spectacle.general import gaussMd

folder = io.path_from_input(argv)
root = io.find_root_folder(folder)
products_gain, results_gain = root/"intermediaries/gain", root/"analysis/gain"
print("Loaded information")

# Get metadata
camera = io.load_metadata(root)
print("Loaded metadata")

colours = camera.bayer_map

nr_files = len(list(folder.glob("*_mean.npy")))
assert nr_files%2 == 0  # equal number for both ISO speeds
nr_each_iso = nr_files//2

isos = np.unique([int(p.stem.split("_")[0]) for p in folder.glob("*_mean.npy")])
low, high = sorted(isos)
_, low_mean  = io.load_npy(folder, f"{low}_*_mean.npy")
_, low_stds  = io.load_npy(folder, f"{low}_*_stds.npy")
print(f"Loaded ISO {low}")
_, high_mean = io.load_npy(folder, f"{high}_*_mean.npy")
_, high_stds = io.load_npy(folder, f"{high}_*_stds.npy")
print(f"Loaded ISO {high}")

low_mean = camera.correct_bias(low_mean )
high_mean = camera.correct_bias(high_mean)

ratio = high_mean / low_mean
q_low, q_high = np.percentile(ratio.ravel(), 0.1), np.percentile(ratio.ravel(), 99.9)

plt.hist(ratio.ravel(), bins=np.linspace(q_low, q_high, 250))
plt.xlabel(f"ISO {high} / ISO {low}")
plt.title(f"Full data set\n$\mu = {ratio.mean():.2f}$, $\sigma = {ratio.std():.2f}$")
plt.show()
plt.close()

ratio_mean = ratio.mean(axis=0)
plt.hist(ratio_mean.ravel(), bins=np.linspace(q_low, q_high, 250))
plt.xlabel(f"ISO {high} / ISO {low}")
plt.title(f"Mean per pixel\n$\mu = {ratio_mean.mean():.2f}$, $\sigma = {ratio_mean.std():.2f}$")
plt.show()
plt.close()

ratio_mean_gauss = gaussMd(ratio_mean, 10)
plt.imshow(ratio_mean_gauss)
plt.colorbar()
plt.title("Mean ratio per pixel; Gauss $\sigma = 10$")
plt.show()
plt.close()

ratio_mean_RGBG,_ = raw.pull_apart(ratio_mean, colours)
ratio_mean_RGBG_gauss = gaussMd(ratio_mean_RGBG, (0,5,5))
vmin, vmax = np.percentile(ratio_mean_RGBG_gauss.ravel(), 0.1), np.percentile(ratio_mean_RGBG_gauss.ravel(), 99.9)
plot.show_RGBG(ratio_mean_RGBG_gauss, vmin=vmin, vmax=vmax)

print(f"RMS noise in images: {np.sqrt(np.mean(low_stds)**2) / np.mean(low_mean) * 100:.1f} %")
print(f"Mean deviation in ratio: {ratio.std()/ratio.mean() * 100:.1f} %")