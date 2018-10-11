import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import plot, io
from phonecal.raw import pull_apart
from phonecal.general import Rsquare, bin_centers
from phonecal.gain import malus, malus_error
from scipy.stats import binned_statistic

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)

angles,  means = io.load_means (folder, retrieve_value=io.split_pol_angle)
angles, jmeans = io.load_jmeans(folder, retrieve_value=io.split_pol_angle)
colours        = io.load_colour(stacks)

offset_angle = np.loadtxt(stacks/"linearity"/"default_angle.dat")
intensities = malus(angles, offset_angle)
intensities_errors = malus_error(angles, offset_angle, sigma_angle0=1, sigma_angle1=1)

means = np.moveaxis(means , 0, 2)
jmeans= np.moveaxis(jmeans, 0, 2)

means_RGBG, _ = pull_apart(means , colours)
jmeans_RGBG, _= pull_apart(jmeans, colours)

middle1, middle2 = means_RGBG.shape[1]//2, means_RGBG.shape[2]//2

#  plot (intensity, DNG value) and (intensity, JPEG value) for the central 4 pixels in the stack
fig, axs = plt.subplots(2, 2, tight_layout=True, figsize=(10,5), sharex=True, sharey=True)
for j in range(4):
    i = j if j < 3 else 1
    c = "rgb"[i]
    ax = axs.ravel()[j]
    ax.errorbar(intensities, jmeans_RGBG[j,middle1,middle2,:,i], xerr=intensities_errors, fmt=f"{c}o")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 255*1.05)

    ax2 = ax.twinx()
    ax2.errorbar(intensities, means_RGBG[j,middle1,middle2], xerr=intensities_errors, fmt="ko")
    ax2.set_ylim(0, 4095*1.05)
    if j%2:
        ax2.set_ylabel("DNG value")
    else:
        ax.set_ylabel("JPEG value")
        ax2.tick_params(axis="y", labelright=False)
    if j//2:
        ax.set_xlabel("Intensity")
fig.savefig(results/"linearity/linearity_DNG_JPEG.png")
plt.close()
print("RGBG JPG-DNG comparison made")

def linear_R2(x, y, saturate=4000):
    ind = np.where(y < saturate)
    p = np.polyfit(x[ind], y[ind], 1)
    pv = np.polyval(p, x[ind])
    R2 = Rsquare(y[ind], pv)
    return R2

print("Doing R^2 comparison...", end=" ")
del jmeans_RGBG
M_reshaped = means_RGBG.reshape(4, -1, means_RGBG.shape[-1])
#M_reshaped = np.ma.array(M_reshaped, mask=M_reshaped>4000)
R2 = np.zeros((4, len(M_reshaped[0])))
for j, M in enumerate(M_reshaped):
    R2[j] = [linear_R2(intensities, row, saturate=4000) for row in M]
    print(j, end=" ")

np.save(results/"linearity/R2.npy", R2)

min_R2 = 0.996
bins = np.linspace(min_R2, 1, 200)
fig, axs = plt.subplots(nrows=3, sharex=True, tight_layout=True, figsize=(6,8), gridspec_kw={"hspace":0, "wspace":0})
R2_RGB = [R2[0], np.concatenate([R2[1], R2[3]]), R2[2]]
for c, ax, R2_C in zip("RGB", axs, R2_RGB):
    ax.hist(R2_C, bins=bins, color=c)
axs[1].set_ylabel("Frequency")
axs[2].set_xlabel("$R^2$")
axs[2].set_xlim(min_R2, 1)
fig.savefig(results/"linearity/R2.png")
plt.close()
print("Made colour plot")

R2R = R2.ravel()

plt.figure(tight_layout=True, figsize=(5,4))
plt.hist(R2R, bins=bins, color='k')
plt.xlabel("$R^2$")
plt.ylabel("Frequency")
plt.xlim(min_R2, 1)
plt.savefig(results/"linearity/R2_combined.png")
plt.close()
print("Made combined plot")

print(f"Lowest: {R2R.min():.5f}")
for percentage in [0.1, 1, 5, 50, 90, 95, 99, 99.9]:
    print(f"{percentage:>5.1f}%: {np.percentile(R2R, percentage):.5f}")
