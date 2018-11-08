import numpy as np
from sys import argv
from phonecal import io
from phonecal.raw import pull_apart
from phonecal.general import Rsquare
from phonecal.gain import malus, malus_error

folder = io.path_from_input(argv)
root, images, stacks, products, results = io.folders(folder)
phone = io.read_json(root/"info.json")

angles,  means = io.load_means (folder, retrieve_value=io.split_pol_angle)
print("Read DNG")
colours        = io.load_colour(stacks)

offset_angle = np.loadtxt(stacks/"linearity"/"default_angle.dat")
print("Read angles")
intensities = malus(angles, offset_angle)
intensities_errors = malus_error(angles, offset_angle, sigma_angle0=1, sigma_angle1=1)

max_value = 2**phone["camera"]["bits"]
saturation = 0.95 * max_value

def linear_R2(x, y, saturate=4000):
    ind = np.where(y < saturate)
    p = np.polyfit(x[ind], y[ind], 1)
    pv = np.polyval(p, x[ind])
    R2 = Rsquare(y[ind], pv)
    return R2

print("Doing R^2 comparison...")

saturated = []
R2 = np.zeros(means.shape[1:])
for i in range(means.shape[1]):
    for j in range(means.shape[2]):
        try:
            R2[i,j] = linear_R2(intensities, means[:,i,j], saturate=saturation)
        except TypeError:  # if fully saturated
            R2[i,j] = np.nan
            saturated.append((i,j))
    if i%5 == 0:
        print(f"{i/means.shape[1]*100:.1f}%")

np.save(products/"linearity_R2.npy", R2)
