import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from phonecal import plot, io
from phonecal.raw import pull_apart
from phonecal.general import Rsquare, bin_centers
from phonecal.gain import malus, malus_error
from glob import glob
from scipy.stats import binned_statistic

folder = argv[1]

angles, means = io.load_means(f"{folder}/stacks/linearity/", retrieve_value=io.split_pol_angle)
angles, jmeans= io.load_jmeans(f"{folder}/stacks/linearity/", retrieve_value=io.split_pol_angle)
colours = np.load(f"{folder}/stacks/colour.npy")

offset_angle = np.loadtxt(f"{folder}/stacks/linearity/default_angle.dat")
intensities = malus(angles, offset_angle)
intensities_errors = malus_error(angles, offset_angle, sigma_angle0=1, sigma_angle1=1)

means = np.moveaxis(means , 0, 2)
jmeans= np.moveaxis(jmeans, 0, 2)

means_RGBG, _ = pull_apart(means , colours)
jmeans_RGBG, _= pull_apart(jmeans, colours)

raise Exception

meanarrs = np.stack(meanarrs)
#vararrs = np.stack(vararrs)
jpgmeanarrs = np.stack(jpgmeanarrs)
#jpgvararrs = np.stack(jpgvararrs)

M_RGBG, _ = pull_apart(meanarrs.T, colors) ; M_RGBG = M_RGBG.T
print("Split DNG means")
J_RGBG, _ = pull_apart(np.moveaxis(jpgmeanarrs, 0, 2), colors)
J_RGBG = np.moveaxis(J_RGBG, 0, 3)
J_RGBG = np.moveaxis(J_RGBG, 2, 0)
print("Split JPEG means")
#V_RGBG, _ = pull_apart(vararrs.T , colors) ; V_RGBG = V_RGBG.T

Is = malus(pols)
Ierrs = malus_error(pols, sigma_angle0=1.0)

m1, m2 = M_RGBG.shape[1]//2, M_RGBG.shape[2]//2
fig, axs = plt.subplots(2, 2, tight_layout=True, figsize=(10,5), sharex=True, sharey=True)
for j in range(4):
    i = j if j < 3 else 1
    c = "rgb"[i]
    ax = axs.ravel()[j]
    ax.errorbar(Is, J_RGBG[:,m1,m2,j,i], xerr=Ierrs, fmt=f"{c}o")
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(0, 255*1.05)

    ax2 = ax.twinx()
    ax2.errorbar(Is, M_RGBG[:,m1,m2,j], xerr=Ierrs, fmt="ko")
    ax2.set_ylim(0, 4095*1.05)
    if j%2:
        ax2.set_ylabel("DNG value")
    else:
        ax.set_ylabel("JPEG value")
        ax2.tick_params(axis="y", labelright=False)
    if j//2:
        ax.set_xlabel("Intensity")
fig.savefig("results/linearity/RGBG.png")
plt.close()
print("RGBG JPG-DNG comparison made")

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
        #stds = binned_statistic(Mc, Jc, statistic=np.std, bins=x).statistic
        lens = binned_statistic(Mc, Jc, statistic="count", bins=x).statistic
        means_all[k,j] = means
        #stds_all[k,j] = stds
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
    plt.savefig(f"results/linearity/dng_jpeg_{label}.png")
    plt.close()

def sRGB(linear, knee, slope, xoff, xmul, yoff, ymul):
    def curve(linear, xoff, xmul, yoff, ymul):
        return yoff + ymul * (xmul * (linear-xoff))**(1/2.4)
    res = curve(linear, xoff, xmul, yoff, ymul)
    linear_offset = curve(knee, xoff, xmul, yoff, ymul) - slope*knee
    res[linear < knee] = linear[linear < knee] * slope + linear_offset
    return res

raise Exception

def linear_R2(Is, row, saturate=4000):
    ind = np.where(row < 4000)
    p = np.polyfit(Is[ind], row[ind], 1)
    pv = np.polyval(p, Is[ind])
    R2 = Rsquare(row[ind], pv)
    return R2

print("R^2 comparison...", end=" ")
M_reshaped = M_RGBG.reshape(len(M_RGBG), -1, 4)
M_reshaped = np.ma.array(M_reshaped, mask=M_reshaped>4000)
R2 = np.zeros((4, len(M_reshaped[0])))
for j in range(4):
    R2[j] = [linear_R2(Is, row, saturate=4000) for row in M_reshaped[...,j].T]
    print(j, end=" ")

fig, axs = plt.subplots(2, 2, sharex=True, sharey=True, tight_layout=True, figsize=(7,7))
for j in range(4):
    ax = axs.ravel()[j]
    c = "rgbg"[j]
    ax.hist(R2[j], bins=np.linspace(0.995,1,200), color=c)
for ax in axs[1]:
    ax.set_xlabel("$R^2$")
for ax in axs[:,0]:
    ax.set_ylabel("Frequency")
fig.savefig("results/linearity/R2.png")
plt.close()
print("made")
