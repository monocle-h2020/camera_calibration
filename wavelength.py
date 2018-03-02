from matplotlib import pyplot as plt
import numpy as np
import rawpy
from cv2 import resize
from scipy.ndimage.filters import gaussian_filter1d as gauss
from sys import argv
import subprocess
import os

filename = argv[1]

RGB = ["r", "g", "b"]
TLpeaks = np.array([436.6, 487.7, 544.45, 611.6])

col0 = 1501
col1 = 1911
col2 = 1949
col3 = 2315

row0 = 2300
row1 = 3750

img = rawpy.imread(filename)
data = img.postprocess(use_camera_wb=True, gamma=(1,1), output_bps=8)
#data_res = resize(data, (data.shape[1]//n, data.shape[0]//n))

thick = data[row0:row1, col0:col1]
thin  = data[row0:row1, col2:col3]

for D in (thick, thin):
    plt.imshow(D)
    plt.show()

def gauss_filter(D, sigma=5, *args, **kwargs):
    """
    Apply a 1-D Gaussian kernel along the wavelength axis
    """
    return gauss(D.astype(float), sigma, *args, axis=0, **kwargs)

thickF = gauss_filter(thick)
thinF  = gauss_filter(thin )

for D in (thickF, thinF):
    plt.imshow(D)
    plt.show()

for j in (0,1,2):
    plt.plot(thick[:, 175, j], c=RGB[j])
plt.xlim(0, thick.shape[0])
plt.ylim(0,255)
plt.show()

for j in (0,1,2):
    plt.plot(thickF[:, 175, j], c=RGB[j])
plt.xlim(0, thickF.shape[0])
plt.ylim(0,255)
plt.show()