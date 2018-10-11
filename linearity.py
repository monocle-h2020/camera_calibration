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

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, tight_layout=True, figsize=(7,7))
for j in range(4):
    ax = axs.ravel()[j]
    c = "rgbg"[j]
    ax.hist(R2[j], bins=np.linspace(0.995,1,200), color=c)
for ax in axs[1]:
    ax.set_xlabel("$R^2$")
for ax in axs[:,0]:
    ax.set_ylabel("Frequency")
fig.savefig(results/"linearity/R2.png")
plt.close()
print("Made colour plot")

R2R = R2.ravel()

plt.figure(tight_layout=True, figsize=(5,4))
plt.hist(R2R, bins=np.linspace(0.997,1,200), color='k')
plt.xlabel("$R^2$")
plt.ylabel("Frequency")
plt.savefig(results/"linearity/R2_combined.png")
plt.close()
print("Made combined plot")

print(f"Lowest: {R2R.min():.5f}")
for percentage in [0.1, 1, 5, 50, 90, 95, 99, 99.9]:
    print(f"{percentage:>5.1f}%: {np.percentile(R2R, percentage):.5f}")

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

raise Exception
