import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import io

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)

r = np.load(products/"linearity_pearson_r.npy")
try:
    r_jpeg = np.load(products/"linearity_pearson_r_jpeg.npy")
except FileNotFoundError:
    jpeg = False
else:
    jpeg = True
r_ravel = r.ravel()
if jpeg:
    r_jpeg_ravel = [r_colour.ravel() for r_colour in r_jpeg]
print("Read Pearson r")

nans = np.where((np.isnan(r_ravel)) | np.isinf(r_ravel))
print(f"Number of megapixels: {len(r_ravel)/1e6:.1f} ; Number of NaN/inf: {len(nans[0])}")
r_ravel = np.delete(r_ravel, nans)
if jpeg:
    for j in range(3):
        nans = np.where((np.isnan(r_jpeg_ravel[j])) | np.isinf(r_jpeg_ravel[j]))
        r_jpeg_ravel[j] = np.delete(r_jpeg_ravel[j], nans)

print("Percentile: RAW r    JPEG_R   JPEG_G   JPEG_B")
print(f"    Lowest: {r_ravel.min():.5f}", end="  ")
if jpeg:
    for j in range(3):
        print(f"{r_jpeg_ravel[j].min():.5f}", end="  ")
print()
for percentage in [0.1, 1, 5, 50, 90, 95, 99, 99.9]:
    print(f"{percentage:>9.1f}%: {np.percentile(r_ravel, percentage):.5f}", end="  ")
    if jpeg:
        for j in range(3):
            print(f"{np.percentile(r_jpeg_ravel[j], percentage):.5f}", end="  ")
    print()

percentile = np.percentile(r_ravel, 0.5)

bins = np.linspace(percentile, 1, 200)
fig, axs = plt.subplots(nrows=3, sharex=True, tight_layout=True, figsize=(6,8), gridspec_kw={"hspace":0, "wspace":0})
r_RGB = [r[0], np.concatenate([r[1], r[3]]), r[2]]
for c, ax, R2_C in zip("RGB", axs, r_RGB):
    ax.hist(R2_C, bins=bins, color=c)
axs[1].set_ylabel("Frequency")
axs[2].set_xlabel("Pearson $r$")
axs[2].set_xlim(percentile, 1)
fig.savefig(results/"linearity/r_RGB.pdf")
plt.close()
print("Made RGB histogram")

plt.figure(tight_layout=True, figsize=(5,4))
plt.hist(r_ravel, bins=bins, color='k')
plt.xlabel("Pearson $r$")
plt.ylabel("Frequency")
plt.xlim(percentile, 1)
plt.xticks(rotation=30)
plt.savefig(results/"linearity/r_linear.pdf")
plt.close()
print("Made linear histogram")

plt.figure(tight_layout=True, figsize=(5,4))
plt.hist(r_ravel, bins=bins, color='k', cumulative=True, density=True)
plt.xlabel("Pearson $r$")
plt.ylabel("Cumulative frequency")
plt.xlim(percentile, 1)
plt.xticks(rotation=30)
plt.ylim(0, 1)
plt.savefig(results/"linearity/r_cumulative.pdf")
plt.close()
print("Made cumulative histogram")

plt.figure(tight_layout=True, figsize=(5,4))
plt.hist(r_ravel, bins=bins, color='k')
plt.xlabel("Pearson $r$")
plt.ylabel("Frequency")
plt.xlim(percentile, 1)
plt.xticks(rotation=30)
plt.yscale("log")
plt.ylim(ymin=0.9)
plt.savefig(results/"linearity/r_log.pdf")
plt.close()
print("Made logarithmic histogram")

if jpeg:
    bins_new = np.linspace(0.9, 1.0, 250)
    plt.figure(tight_layout=True, figsize=(5,4))
    plt.hist(r_ravel, bins=bins_new, color='k', density=True)
    for j, c in enumerate("rgb"):
        plt.hist(r_jpeg_ravel[j], bins=bins_new, color=c, alpha=0.7, density=True)
    plt.xlabel("Pearson $r$")
    plt.ylabel("Density")
    plt.yscale("log")
    plt.xlim(0.9, 1)
    plt.xticks(rotation=30)
    plt.savefig(results/"linearity/r_raw_jpeg.pdf")
    plt.close()
    print("Made RAW/JPEG histogram")
