import numpy as np
import rawpy
from sys import argv
from matplotlib import pyplot as plt
from ispex.general import gauss_filter
from ispex import plot

filename = argv[1]

def pull_apart(raw_img):
    R = raw_img[1::2,1::2]
    G = raw_img[ ::2,1::2]
    B = raw_img[ ::2, ::2]
    G2= raw_img[1::2, ::2]
    return R, G, B, G2

def gauss_raw(raw_img, **kwargs):
    """
    Assuming
    BGBG
    GRGR
    BGBG
    GRGR
    """
    gaussed = raw_img.copy()
    R, G, B, G2 = pull_apart(raw_img)
    Rg = gauss_filter(R, **kwargs)
    Gg = gauss_filter(G, **kwargs)
    Bg = gauss_filter(B, **kwargs)
    G2g= gauss_filter(G2,**kwargs)
    gaussed[1::2,1::2] = Rg
    gaussed[ ::2,1::2] = Gg
    gaussed[ ::2, ::2] = Bg
    gaussed[1::2, ::2] = G2g
    return gaussed

img = rawpy.imread(filename)
imgarray = img.raw_image.astype(np.int16)
R, G, B, G2 = pull_apart(imgarray)

for color, arr in zip(['r', 'g', 'b', 'y'], [R,G,B,G2]):
    plt.plot(arr[600:750].mean(0), c=color)
plt.xlim(1150, 1700)
plt.ylim(500, 1000)
plt.show()
plt.close()

plot.RGBG(R, G, B, G2, vmax=800, saveto="RGBG_split.png", size=15)
imgarrayg = gauss_raw(imgarray, sigma=250)
Rg, Gg, Bg, G2g = pull_apart(imgarrayg)
plot.RGBG(Rg, Gg, Bg, G2g, vmax=800, saveto="RGBG_split_gauss.png", size=15)