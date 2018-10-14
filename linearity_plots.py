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

means_to_plot  = np.zeros((4, means.shape[0]))  # in RGBG order
jmeans_to_plot = np.zeros((4, means.shape[0]))
middle1, middle2 = means.shape[1]//2, means.shape[2]//2
index_pairs = [(middle1, middle2), (middle1+1, middle2), (middle1, middle2+1), (middle1+1,middle2+1)]
for pair in index_pairs:
    i1, i2 = pair
    colour_index = colours[i1, i2]
    jcolour_index = colour_index if colour_index < 3 else 1
    means_to_plot [colour_index] = means [:,i1,i2]
    jmeans_to_plot[colour_index] = jmeans[:,i1,i2,jcolour_index]
print("Found pixels to plot")

#  plot (intensity, DNG value) and (intensity, JPEG value) for the central 4 pixels in the stack
fig, axs = plt.subplots(2, 2, tight_layout=True, figsize=(10,5), sharex=True, sharey=True)
for j in range(4):
    i = j if j < 3 else 1
    c = "rgb"[i]
    ax = axs.ravel()[j]
    jmean = jmeans_to_plot[j]
    ax.errorbar(intensities, jmean, xerr=intensities_errors, fmt=f"{c}o")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 255*1.05)

    mean = means_to_plot[j]
    ax2 = ax.twinx()
    ax2.errorbar(intensities, mean, xerr=intensities_errors, fmt="ko")
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
