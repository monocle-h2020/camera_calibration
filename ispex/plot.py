import numpy as np
from matplotlib import pyplot as plt, patheffects as pe
from .raw import split_RGBG, x as x_raw, range_y

def _saveshow(saveto=None, close=True, **kwargs):
    if saveto is None:
        plt.show()
    else:
        plt.savefig(saveto, **kwargs)
    if close:
        plt.close()

def _rgbplot(x, y, func=plt.plot, **kwargs):
    RGBY = ["R", "G", "B", "Y"]
    for j in range(4):
        func(x, y[..., j], c=RGBY[j], **kwargs)

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

def RGBG(RGBG, saveto=None, size=13, **kwargs):
    R, G, B, G2 = split_RGBG(RGBG)
    fig, axs = plt.subplots(2,2,sharex=True,sharey=True,figsize=(size,size*0.77))
    axs[0,0].imshow(B,  cmap=plt.cm.Blues_r , **kwargs)
    axs[0,1].imshow(G,  cmap=plt.cm.Greens_r, **kwargs)
    axs[1,0].imshow(G2, cmap=plt.cm.Greens_r, **kwargs)
    axs[1,1].imshow(R,  cmap=plt.cm.Reds_r  , **kwargs)
    for ax in axs.ravel():
        ax.axis("off")
    fig.subplots_adjust(hspace=.001, wspace=.001)
    _saveshow(saveto, transparent=True)

def _to_8_bit(data, maxvalue=4096, boost=1):
    converted = (data / maxvalue * 255).astype(np.uint8)
    converted = boost * (converted - 30)
    converted[converted > 255] = 255
    return converted

def RGBG_stacked(RGBG, maxvalue=4096, saveto=None, size=13, boost=1, **kwargs):
    """
    Ignore G2 for now
    """
    plt.figure(figsize=(size,size))
    to_plot = _to_8_bit(RGBG[:,:,:3], maxvalue=maxvalue, boost=boost)
    plt.imshow(to_plot, **kwargs)
    plt.axis("off")
    _saveshow(saveto, transparent=True)

def RGBG_stacked_with_graph(RGBG, x=x_raw, yrange=range_y, maxvalue=4096, boost=5, saveto=None, **kwargs):
    R, G, B, G2 = split_RGBG(RGBG)  # change to RGBG
    stacked = np.dstack((R, (G+G2)/2, B))
    stacked_8_bit = _to_8_bit(stacked, maxvalue=maxvalue, boost=boost)

    fig, ax1 = plt.subplots(figsize=(17,5))
    ax1.imshow(stacked_8_bit, extent=(x[0], x[-1], yrange[1], yrange[0]))
    ax1.set_xlabel("Pixel")
    ax1.set_ylabel("Pixel")
    ax1.set_ylim(yrange[1], yrange[0])

    ax2 = ax1.twinx()
    p_eff = [pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]
    meaned = RGBG.mean(axis=0)
    _rgbplot(x, meaned, func=ax2.plot, path_effects = p_eff)  # change to RGBG
    ax2.set_ylabel("Mean RGBG value")
    ax2.set_xlim(x[0], x[-1])
    ax2.set_ylim(meaned.min()*0.99, meaned.max()*1.01)

    fig.tight_layout()
    _saveshow(saveto, transparent=True)
