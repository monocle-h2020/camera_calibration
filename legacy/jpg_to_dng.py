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

offset_angle = io.load_angle(stacks)
intensities = malus(angles, offset_angle)
intensities_errors = malus_error(angles, offset_angle, sigma_angle0=1, sigma_angle1=1)

means = np.moveaxis(means , 0, 2)
jmeans= np.moveaxis(jmeans, 0, 2)

means_RGBG, _ = pull_apart(means , colours)
jmeans_RGBG, _= pull_apart(jmeans, colours)

raise Exception

maxval = 4096
x = np.arange(0.5, maxval+1.5, 1)
means_all = np.zeros((len(M_RGBG), 4, maxval))
stds_all = np.zeros((len(M_RGBG), 4, maxval))
lens_all = np.zeros((len(M_RGBG), 4, maxval))
for k, (M, J) in enumerate(zip(M_RGBG, J_RGBG)):
    for j in range(4):
        if j <= 2:
            i = j
        else:
            i = 1
        Mc = M[...,j].ravel()
        Jc = J[...,j,i].ravel()
        means, bin_edges, bin_number = binned_statistic(Mc, Jc, statistic="mean", bins=x)
        bc = bin_centers(bin_edges)
        lens = binned_statistic(Mc, Jc, statistic="count", bins=x).statistic
        means_all[k,j] = means
        lens_all[k,j] = lens
    print(k, end=" ")
print("")

x = np.arange(maxval)
for j in range(4):
    c = "rgbg"[j]
    label = c if j < 3 else "g2"
    plt.figure(figsize=(10,7), tight_layout=True)
    for m,s in zip(means_all[:,j], stds_all[:,j]):
        plt.scatter(bc, m, s=1, c=c)
        #plt.fill_between(bc, m-s, m+s, alpha=0.5, color=c)
        plt.xlim(0, maxval*1.03)
        plt.ylim(0, 260)
        plt.xlabel("DNG value")
        plt.ylabel("JPEG value")
    plt.savefig(results/f"linearity/dng_jpeg_{label}.png")
    plt.close()

def sRGB(linear, knee, slope, xoff, xmul, yoff, ymul):
    def curve(linear, xoff, xmul, yoff, ymul):
        return yoff + ymul * (xmul * (linear-xoff))**(1/2.4)
    res = curve(linear, xoff, xmul, yoff, ymul)
    linear_offset = curve(knee, xoff, xmul, yoff, ymul) - slope*knee
    res[linear < knee] = linear[linear < knee] * slope + linear_offset
    return res
