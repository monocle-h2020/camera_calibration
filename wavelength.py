from matplotlib import pyplot as plt
import numpy as np
import rawpy
from cv2 import resize
from scipy.ndimage.filters import gaussian_filter1d as gauss
from sys import argv

filename = argv[1]

TLpeaks = np.array([436.6, 487.7, 544.45, 611.6])

col0 = 1512
col1 = 2315

row0 = 2300
row1 = 3750

img = rawpy.imread(filename)
data = img.postprocess(use_camera_wb=True, gamma=(1,1), output_bps=8)
#data_res = resize(data, (data.shape[1]//n, data.shape[0]//n))

crop = data[row0:row1, col0:col1]
plt.imshow(crop)
plt.show()

filtered = gauss(crop.astype(float), 5, axis=0)

for x in (200, 600):
    RGB = ["r", "g", "b"]
    for j in (0,1,2):
        plt.plot(filtered[:,x,j], c=RGB[j])
    plt.xlim(0, crop.shape[0])
    plt.ylim(0, 255)
    plt.show()