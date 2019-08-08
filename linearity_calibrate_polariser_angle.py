import numpy as np
from sys import argv
from matplotlib import pyplot as plt
from spectacle import io, raw
from spectacle.gain import malus
from scipy.optimize import curve_fit

folder = argv[1]

angles, means = io.load_means(f"{folder}/stacks/linearity/", retrieve_value=io.split_pol_angle)
angles, stds  = io.load_stds (f"{folder}/stacks/linearity/", retrieve_value=io.split_pol_angle)
colours = np.load(f"{folder}/stacks/colour.npy")

mean_reshaped = np.moveaxis(means, 0, 2)
stds_reshaped = np.moveaxis(stds , 0, 2)

def malus_amp(angles, amplitude, offset_angle, offset_intensity):
    I = offset_intensity + amplitude * malus(angles, offset_angle)
    return I

def fit(data):
    popt, pcov = curve_fit(malus_amp, angles, data, p0=[1000,74,1000])
    return popt[1]

meanRGBG, offsets = raw.pull_apart(mean_reshaped, colours)
stdsRGBG, offsets = raw.pull_apart(stds_reshaped, colours)

x = np.arange(means.shape[2])
y = np.arange(means.shape[1])
X, Y = np.meshgrid(x, y)
(x0, y0) = (len(x) / 2, len(y) / 2)
D = np.sqrt((X - x0)**2 + (Y - y0)**2)
D_split, offsets = raw.pull_apart(D, colours)

outer_radii = np.arange(1000, 2000, 75)

def ring_mean(outer_radius, data, distances, width=10):
    ind = np.where((distances < outer_radius) & (distances <= outer_radius - width))
    mean = np.mean(data[ind])
    std  = np.std (data[ind])
    return mean, std

allmean = np.zeros((4, len(outer_radii), len(angles)))
allstds = allmean.copy()

for j in range(4):
    for i, radius in enumerate(outer_radii):
        for k, angle in enumerate(angles):
            allmean[j,i,k], allstds[j,i,k] = ring_mean(radius, meanRGBG[j,...,k], D_split[j])
    print(j)

ringmeans = np.zeros(len(outer_radii))
ringstds  = ringmeans.copy()
for x, (ring_mean, ring_std) in enumerate(zip(allmean[2], allstds[2])):
    popt, pcov = curve_fit(malus_amp, angles, ring_mean, sigma=ring_std, p0=[1500,74,528])
    ringmeans[x] = popt[1]
    ringstds [x] = np.sqrt(pcov[1,1])

angle = np.average(ringmeans, weights=1/ringstds)
np.savetxt(f"{folder}/stacks/linearity/default_angle.dat", np.array([angle]))