from matplotlib import pyplot as plt
import numpy as np
import rawpy
from cv2 import resize
from sys import argv

filename = argv[1]
n = int(argv[2])

img = rawpy.imread(filename)
data = img.postprocess(use_camera_wb=True, gamma=(1,1), output_bps=8)

RGB = ["r", "g", "b"]
for j in (0,1,2):
    plt.plot(data[:, 1700, j], c=RGB[j])
plt.xlim(0, data.shape[0])
plt.ylim(0, 255)
plt.show()

data_res = resize(data, (data.shape[1]//n, data.shape[0]//n))
for j in (0,1,2):
    plt.plot(data_res[:, 1700//n, j], c=RGB[j])
plt.xlim(0, data_res.shape[0])
plt.ylim(0, 255)
plt.show()