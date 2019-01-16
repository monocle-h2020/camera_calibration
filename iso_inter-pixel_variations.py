import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import raw, io, plot
from phonecal.general import gaussMd

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")

products_gain, results_gain = products/"gain", results/"gain"
print("Loaded information")

colours      = io.load_colour(stacks  )
bias         = io.load_bias  (products)

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

low_mean  -= bias
high_mean -= bias

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