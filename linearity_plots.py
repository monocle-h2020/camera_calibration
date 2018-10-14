import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import io
from phonecal.raw import pull_apart
from phonecal.gain import malus, malus_error

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)

angles,  means = io.load_means (folder, retrieve_value=io.split_pol_angle)
print("Read DNG")
angles, jmeans = io.load_jmeans(folder, retrieve_value=io.split_pol_angle)
print("Read JPEG")
colours        = io.load_colour(stacks)

offset_angle = np.loadtxt(stacks/"linearity"/"default_angle.dat")
print("Read angles")
intensities = malus(angles, offset_angle)
intensities_errors = malus_error(angles, offset_angle, sigma_angle0=1, sigma_angle1=1)

means = np.moveaxis(means , 0, 2)
jmeans= np.moveaxis(jmeans, 0, 2)
print("Reshaped arrays")

means_RGBG, _ = pull_apart(means , colours)
jmeans_RGBG, _= pull_apart(jmeans, colours)
print("Reshaped to RGBG")

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
    ax2.set_ylim(0, 1024*1.05)
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
