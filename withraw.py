import numpy as np
import rawpy
from sys import argv
from matplotlib import pyplot as plt, patheffects as pe
from ispex.general import gauss_filter
from ispex import plot

filename = argv[1]
handle = filename.split()

def _find_offset(color_pattern, colour):
    pos = np.array(np.where(color_pattern == colour)).T[0]
    return pos

def pull_apart(raw_img, color_pattern, color_desc=b"RGBG"):
    if color_desc != b"RGBG":
        raise ValueError(f"Image is of type {raw_img.color_desc} instead of RGBG")
    offsets = np.array([_find_offset(color_pattern, i) for i in range(4)])
    offX, offY = offsets.T
    R, G, B, G2 = [raw_img[x::2, y::2] for x, y in zip(offX, offY)]
    RGBG = np.dstack((R, G, B, G2))
    return RGBG, offsets

def split_RGBG(RGBG):
    R, G, B, G2 = RGBG.T
    R, G, B, G2 = R.T, G.T, B.T, G2.T
    return R, G, B, G2

def put_together(R, G, B, G2, offsets):
    result = np.zeros((R.shape[0]*2, R.shape[1]*2))
    for colour, offset in zip([R,G,B,G2], offsets):
        x, y = offset
        result[x::2, y::2] = colour
    result = result.astype(R.dtype)
    return result

img = rawpy.imread(filename)
imgarray = img.raw_image.astype(np.int16)
RGBG, offsets = pull_apart(imgarray, img.raw_pattern)
R, G, B, G2 = split_RGBG(RGBG)

stacked = np.dstack([R,(G+G2)/2,B])/4096*255
plt.figure(figsize=(20,20))
plt.imshow(stacked.astype(np.uint8))
plt.axis("off")
plt.tight_layout(True)
plt.savefig("RGBG_stacked.png", transparent=True)
plt.close()

cutout = imgarray[600:1600, 2300:3500]
plt.figure(figsize=(10,7))
for color, arr in zip(['r', 'g', 'b', 'y'], [R,G,B,G2]):
    plt.plot(arr[600:750].mean(0), c=color)
plt.xlim(1000, 1750)
plt.ylim(cutout.min()*0.99, cutout.max()*1.01)
plt.show()
plt.close()

ymin, ymax = 550, 750
xmin, xmax = 1120, 1750
plt.figure(figsize=(17,5))
brighter = 5 * stacked - 150
brighter[brighter > 255] = 255
plt.imshow(brighter.astype(np.uint8))
minval = np.dstack([R,G,B,G2])[600:750,xmin:xmax].min()
maxval = np.dstack([R,G,B,G2])[600:750,xmin:xmax].max()
for color, arr in zip(['r', 'g', 'b', 'y'], [R,G,B,G2]):
    line = arr[600:750].mean(0)
    line -= minval
    line /= (maxval-minval)
    line *= (ymax - ymin)
    line = ymax - line
    plt.plot(line, c=color, path_effects=[pe.Stroke(linewidth=5, foreground='k'), pe.Normal()])
plt.ylim(ymax, ymin)
plt.xlim(xmin, xmax)
plt.xlabel("Pixel")
plt.ylabel("Pixel")
plt.title("iSPEX spectrum with normalised mean curves")
plt.savefig("RGBG_spectrum.png", transparent=True)
plt.close()

plot.RGBG(R, G, B, G2, vmax=800, saveto="RGBG_split.png", size=30)
RGBGg = gauss_filter(RGBG)
Rg, Gg, Bg, G2g = split_RGBG(RGBGg)
plot.RGBG(Rg, Gg, Bg, G2g, vmax=800, saveto="RGBG_split_gauss.png", size=30)