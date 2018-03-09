import numpy as np
from matplotlib import pyplot as plt

def _rgbplot(x, y, func=plt.plot, **kwargs):
    RGB = ["R", "G", "B"]
    for j in (0,1,2):
        func(x, y[..., j], c=RGB[j], **kwargs)

def plot_photo(data, saveto=None, **kwargs):
    plt.imshow(data.astype("uint8"), **kwargs)
    plt.xlabel("$y$")
    plt.ylabel("$x$")
    if saveto is not None:
        plt.savefig(saveto)
    else:
        plt.show()