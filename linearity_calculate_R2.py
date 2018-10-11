import numpy as np
from sys import argv
from phonecal import io
from phonecal.raw import pull_apart
from phonecal.general import Rsquare
from phonecal.gain import malus, malus_error

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

def linear_R2(x, y, saturate=4000):
    ind = np.where(y < saturate)
    p = np.polyfit(x[ind], y[ind], 1)
    pv = np.polyval(p, x[ind])
    R2 = Rsquare(y[ind], pv)
    return R2

print("Doing R^2 comparison...", end=" ")

M_reshaped = means_RGBG.reshape(4, -1, means_RGBG.shape[-1])
#M_reshaped = np.ma.array(M_reshaped, mask=M_reshaped>4000)
R2 = np.zeros((4, len(M_reshaped[0])))
for j, M in enumerate(M_reshaped):
    R2[j] = [linear_R2(intensities, row, saturate=4000) for row in M]
    print(j, end=" ")

np.save(products/"linearity_R2.npy", R2)
