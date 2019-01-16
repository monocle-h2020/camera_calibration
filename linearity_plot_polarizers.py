import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import io
from phonecal.linearity import malus, malus_error

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")

colours = io.load_colour(stacks)

array_size = io.array_size_dng(folder)
mid1, mid2 = array_size // 2
center = np.s_[mid1:mid1+2, mid2:mid2+2]

colours_here = colours[center].ravel()

angles, means = io.load_means (folder, retrieve_value=io.split_pol_angle, selection=center)
means = means.reshape((len(means), -1))
print("Loaded DNG data")

angles, jmeans = io.load_jmeans(folder, retrieve_value=io.split_pol_angle, selection=center)
jmeans = jmeans.reshape((len(jmeans), -1, 3))

offset_angle = io.load_angle(stacks)
print("Read angles")
intensities = malus(angles, offset_angle)
intensities_errors = malus_error(angles, offset_angle, sigma_angle0=1, sigma_angle1=1)

max_value = 2**phone["camera"]["bits"]

for j in range(4):
    colour_index = colours_here[j]
    colour = "rgbg"[colour_index]
    if colour_index < 3:
        i = colour_index
        label = colour
    else:
        i = 1
        label = "g2"
    mean_dng =  means[:, j]
    mean_jpg = jmeans[:, j, i]

    fig, ax = plt.subplots(1, 1, figsize=(3.3,2), tight_layout=True)
    ax.errorbar(intensities, mean_jpg, xerr=intensities_errors, fmt=f"{colour}o", ms=3)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 255*1.02)
    ax.set_xticks(np.arange(0,1.2,0.2))
    ax.set_yticks(np.arange(0, 255, 50))
    ax2 = ax.twinx()
    ax2.errorbar(intensities, mean_dng, xerr=intensities_errors, fmt="ko", ms=3)
    ax2.set_ylim(0, max_value*1.02)
    ax2.locator_params(axis="y", nbins=5)
    jpeglabel = ax.set_ylabel("JPEG value")
    jpeglabel.set_color(colour)
    ax.tick_params(axis="y", colors=colour)
    ax2.set_ylabel("RAW value")
    ax.set_xlabel("Relative incident intensity")
    fig.savefig(results/f"linearity/linearity_DNG_JPEG_singlepixel_{label}.pdf")
    plt.close()
    print(f"Plotted pixel {j} ({label})")
