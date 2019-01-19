import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import io

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)

R2 = np.load(products/"linearity_R2.npy")
R2R = R2.ravel()
print("Read R^2")

nans = np.where((np.isnan(R2R)) | np.isinf(R2R))
print(f"Number of megapixels: {len(R2R)/1e6:.1f} ; Number of NaN/inf: {len(nans[0])}")

R2R = np.delete(R2R, nans)

print(f"Lowest: {R2R.min():.5f}")
for percentage in [0.1, 1, 5, 50, 90, 95, 99, 99.9]:
    print(f"{percentage:>5.1f}%: {np.percentile(R2R, percentage):.5f}")

percentile = np.percentile(R2R, 0.5)

bins = np.linspace(percentile, 1, 200)
fig, axs = plt.subplots(nrows=3, sharex=True, tight_layout=True, figsize=(6,8), gridspec_kw={"hspace":0, "wspace":0})
R2_RGB = [R2[0], np.concatenate([R2[1], R2[3]]), R2[2]]
for c, ax, R2_C in zip("RGB", axs, R2_RGB):
    ax.hist(R2_C, bins=bins, color=c)
axs[1].set_ylabel("Frequency")
axs[2].set_xlabel("$R^2$")
axs[2].set_xlim(percentile, 1)
fig.savefig(results/"linearity/R2_RGB.pdf")
plt.close()
print("Made RGB histogram")

plt.figure(tight_layout=True, figsize=(5,4))
plt.hist(R2R, bins=bins, color='k')
plt.xlabel("$R^2$")
plt.ylabel("Frequency")
plt.xlim(percentile, 1)
plt.savefig(results/"linearity/R2_linear.pdf")
plt.close()
print("Made linear histogram")

plt.figure(tight_layout=True, figsize=(5,4))
plt.hist(R2R, bins=bins, color='k', cumulative=True, density=True)
plt.xlabel("$R^2$")
plt.ylabel("Cumulative frequency")
plt.xlim(percentile, 1)
plt.ylim(0, 1)
plt.savefig(results/"linearity/R2_cumulative.pdf")
plt.close()
print("Made cumulative histogram")

plt.figure(tight_layout=True, figsize=(5,4))
plt.hist(R2R, bins=bins, color='k')
plt.xlabel("$R^2$")
plt.ylabel("Frequency")
plt.xlim(percentile, 1)
plt.yscale("log")
plt.ylim(ymin=0.9)
plt.savefig(results/"linearity/R2_log.pdf")
plt.close()
print("Made logarithmic histogram")
