import numpy as np
from matplotlib import pyplot as plt

def _saveshow(saveto=None, close=True, **kwargs):
    if saveto is None:
        plt.show()
    else:
        plt.savefig(saveto, **kwargs)
    if close:
        plt.close()

def _rgbplot(x, y, func=plt.plot, **kwargs):
    RGB = ["R", "G", "B"]
    for j in (0,1,2):
        func(x, y[..., j], c=RGB[j], **kwargs)

def plot_spectrum(x, y, saveto=None, ylabel="$C$", xlabel="$\lambda$ (nm)", **kwargs):
    plt.figure(figsize=(10, 5))
    _rgbplot(x, y)
    plt.axis("tight")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    try:
        plt.gca().set(**kwargs)
    except:
        pass
    _saveshow(saveto)

def plot_photo(data, saveto=None, **kwargs):
    plt.imshow(data.astype("uint8"), **kwargs)
    plt.xlabel("$y$")
    plt.ylabel("$x$")
    _saveshow(saveto)

def fluorescent_lines(y, lines, lines_fit, saveto=None):
    _rgbplot(y, lines, func=plt.scatter, alpha=0.03)
    _rgbplot(y, lines_fit, ls="--")
    plt.title("Locations of RGB maxima")
    plt.xlabel("$y$")
    plt.ylabel("$x_{peak}$")
    _saveshow(saveto)

def _wavelength_coefficients_single(y, coefficients, coefficients_fit, nr=0, saveto=None):
    plt.scatter(y, coefficients, c='r')
    plt.plot(y, coefficients_fit, c='k', lw=3)
    plt.xlim(y[0], y[-1])
    plt.ylim(coefficients.min(), coefficients.max())
    plt.title(f"Coefficient {nr} of wavelength fit")
    plt.xlabel("$y$")
    plt.ylabel(f"$p_{nr}$")
    _saveshow(saveto)

def wavelength_coefficients(y, coefficients, coefficients_fit, saveto=None):
    for j in range(coefficients_fit.shape[1]):
        _wavelength_coefficients_single(y, coefficients[:,j], coefficients_fit[:,j], nr=j, saveto=saveto)

def histogram(data, saveto=None, **kwargs):
    counts = np.bincount(data.flatten())
    plt.scatter(np.arange(len(counts)), counts, **kwargs)
    plt.yscale("log")
    plt.ylim(ymin=0.9)
    plt.xlim(0, len(counts)*1.01)
    plt.xlabel("RGB value")
    plt.ylabel("Number of pixels")
    plt.tight_layout(True)
    _saveshow(saveto)

def RGBG(R, G, B, G2, saveto=None, size=13, **kwargs):
    fig, axs = plt.subplots(2,2,sharex=True,sharey=True,figsize=(size,size*0.77))
    axs[0,0].imshow(B,  cmap=plt.cm.Blues_r , **kwargs)
    axs[0,1].imshow(G,  cmap=plt.cm.Greens_r, **kwargs)
    axs[1,0].imshow(G2, cmap=plt.cm.Greens_r, **kwargs)
    axs[1,1].imshow(R,  cmap=plt.cm.Reds_r  , **kwargs)
    for ax in axs.ravel():
        ax.axis("off")
    fig.subplots_adjust(hspace=.001, wspace=.001)
    _saveshow(saveto, transparent=True)