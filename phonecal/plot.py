import numpy as np
from matplotlib import pyplot as plt, patheffects as pe, ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from . import raw

cmaps = {"R": plt.cm.Reds, "G": plt.cm.Greens, "B": plt.cm.Blues,
         "Rr": plt.cm.Reds_r, "Gr": plt.cm.Greens_r, "Br": plt.cm.Blues_r,
         None: plt.cm.viridis}

def _saveshow(saveto=None, close=True, **kwargs):
    if saveto is None:
        plt.show()
    else:
        plt.savefig(saveto, **kwargs)
    if close:
        plt.close()

def _rgbplot(x, y, func=plt.plot, **kwargs):
    RGB = ["R", "G", "B"]
    for j in range(3):
        func(x, y[j], c=RGB[j], **kwargs)

def _rgbgplot(x, y, func=plt.plot, **kwargs):
    RGBY = ["R", "G", "B", "Y"]
    for j in range(4):
        func(x, y[j], c=RGBY[j], **kwargs)

def plot_spectrum(x, y, saveto=None, ylabel="$C$", xlabel="$\lambda$ (nm)", **kwargs):
    plt.figure(figsize=(10, 5), tight_layout=True)
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

def imshow_tight(data, figsize=(15,15), saveto=None, **kwargs):
    plt.figure(figsize=figsize, tight_layout=True)
    plt.imshow(data, **kwargs)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    _saveshow(saveto, transparent=True)

def bitmap(data, dpi=96, saveto=None, **kwargs):
    plt.figure(figsize=(data.shape[1]/dpi, data.shape[0]/dpi), dpi=dpi, tight_layout=True)
    plt.imshow(data, interpolation="none", aspect="equal", **kwargs)
    plt.axis("off")
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    _saveshow(saveto, transparent=True, dpi=dpi)

def fluorescent_lines(y, lines, lines_fit, saveto=None):
    plt.figure(figsize=(10, 5))
    _rgbplot(y, lines, func=plt.scatter, s=25)
    p_eff = [pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]
    _rgbplot(y, lines_fit, path_effects=p_eff)
    plt.title("Locations of RGB maxima")
    plt.xlabel("$y$")
    plt.ylabel("$x_{peak}$")
    plt.axis("tight")
    plt.tight_layout()
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
        try:
            # save every coefficient plot in its own file
            # TODO: make one plot with shared x axis
            saveto1 = saveto.split(".")
            saveto1 = saveto1[0] + "_" + str(j) + "." + saveto1[1]
        except AttributeError:
            saveto1 = saveto
        _wavelength_coefficients_single(y, coefficients[:,j], coefficients_fit[:,j], nr=j, saveto=saveto1)

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
    R, G, B, G2 = raw.split_RGBG(RGBG)
    frac = RGBG.shape[1]/RGBG.shape[2]
    fig, axs = plt.subplots(2,2,sharex=True,sharey=True,figsize=(size,size*frac))
    axs[0,0].imshow(B,  cmap=plt.cm.Blues_r , aspect="equal", **kwargs)
    axs[0,1].imshow(G,  cmap=plt.cm.Greens_r, aspect="equal", **kwargs)
    axs[1,0].imshow(G2, cmap=plt.cm.Greens_r, aspect="equal", **kwargs)
    axs[1,1].imshow(R,  cmap=plt.cm.Reds_r  , aspect="equal", **kwargs)
    for ax in axs.ravel():
        ax.axis("off")
    fig.subplots_adjust(hspace=.001, wspace=.001)
    _saveshow(saveto, transparent=True)

def _to_8_bit(data, maxvalue=4096, boost=1):
    converted = data.astype(np.float) / maxvalue * 255
    converted = boost * converted - (boost-1) * 30
    converted[converted > 255] = 255  # -> np.clip
    converted[converted < 0]   = 0
    converted = converted.astype(np.uint8)
    return converted

def RGBG_stacked(RGBG, maxvalue=4096, saveto=None, size=13, boost=5, xlabel="Pixel $x$", show_axes=False, **kwargs):
    """
    Ignore G2 for now
    """
    plt.figure(figsize=(size,size))
    to_plot = _to_8_bit(RGBG[:3], maxvalue=maxvalue, boost=boost)
    to_plot = np.moveaxis(to_plot, 0, 2)
    plt.imshow(to_plot, **kwargs)
    plt.xlabel(xlabel)
    plt.ylabel("Pixel $y$")
    if not show_axes:
        plt.axis("off")
    _saveshow(saveto, transparent=True)

def RGBG_stacked_with_graph(RGBG, x=raw.x, maxvalue=4096, boost=5, saveto=None, xlabel="Pixel $x$", **kwargs):
    R, G, B, G2 = raw.split_RGBG(RGBG)  # change to RGBG
    stacked = np.stack((R, (G+G2)/2, B))
    stacked_8_bit = _to_8_bit(stacked, maxvalue=maxvalue, boost=boost)
    stacked_8_bit = np.moveaxis(stacked_8_bit, 0, 2)
    stacked_8_bit = stacked_8_bit.swapaxes(0,1)

    fig, ax1 = plt.subplots(figsize=(17,5))
    ax1.imshow(stacked_8_bit, **kwargs)
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Pixel $y$")
    ax1.set_ylim(raw.ymax, raw.ymin)

    ax2 = ax1.twinx()
    p_eff = [pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]
    meaned = RGBG.mean(axis=2)
    _rgbplot(x, meaned, func=ax2.plot, path_effects = p_eff)  # change to RGBG
    ax2.set_ylabel("Mean RGBG value")
    ax2.set_xlim(x[0], x[-1])
    ax2.set_ylim(meaned.min()*0.99, meaned.max()*1.01)

    fig.tight_layout()
    _saveshow(saveto, transparent=True)

def Bayer(RGB_array, size=10, saveto=None, **kwargs):
    plt.figure(figsize=(size, size))
    plt.imshow(RGB_array, aspect="equal", interpolation="none", **kwargs)
    plt.axis("off")
    _saveshow(saveto, transparent=True, dpi=900)

def hexbin_colour(colour, *args, **kwargs):
    plt.hexbin(*args, cmap=cmaps[colour], **kwargs)

def linearity(I, C, colour, gridsize=250, bins="log", extent=(0,1,0,4095), saveto=None, **kwargs):
    plt.figure(figsize=(10, 7))
    hexbin_colour(colour, I, C, gridsize=gridsize, bins=bins, extent=extent, **kwargs)
    plt.xlabel("Intensity")
    plt.ylabel(colour + " value")
    plt.xlim(*extent[:2])
    plt.ylim(*extent[2:])
    _saveshow(saveto)

def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)

def show_image(data, colour=None, colorbar_label="", saveto=None, **kwargs):
    cmap = cmaps[colour+"r"] if colour else plt.cm.viridis
    plt.figure(figsize=(3.3,3*data.shape[0]/data.shape[1]), tight_layout=True)
    img = plt.imshow(data, cmap=cmap, **kwargs)
    plt.xticks([])
    plt.yticks([])
    colorbar_here = colorbar(img)
    colorbar_here.set_label(colorbar_label)
    colorbar_here.locator = ticker.MaxNLocator(nbins=5)
    colorbar_here.update_ticks()
    _saveshow(saveto)
    
def show_RGBG(data, colour=None, colorbar_label="", saveto=None, **kwargs):
    fig, axs = plt.subplots(nrows=4, sharex=True, sharey=True, figsize=(3.3,5), squeeze=True, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0})
    for ax, data_c, c in zip(axs, data, "RGBG"):
        img = ax.imshow(data_c, cmap=cmaps[c+"r"], **kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        colorbar_here = colorbar(img)
        if ax is axs[2]:
            colorbar_here.set_label(colorbar_label)
        colorbar_here.locator = ticker.MaxNLocator(nbins=4)
        colorbar_here.update_ticks()
    _saveshow(saveto)

def hist_bias_ron(data, xlim=(0, 30), nrbins=500, xlabel="Bias (ADU)", saveto=None):
    plt.figure(figsize=(10,7), tight_layout=True)
    plt.hist(data.ravel(), bins=np.linspace(*xlim, nrbins), color='k')
    plt.xlabel(xlabel)
    plt.xlim(*xlim)
    plt.ylim(0.9, data.size)
    plt.yscale("log")
    plt.ylabel("Frequency")
    plt.grid(ls="--")
    _saveshow(saveto)

def hist_bias_ron_kRGB(data_RGBG, xlim=(0, 30), nrbins=500, xlabel="Bias (ADU)", saveto=None):
    data_KRGB = [data_RGBG.ravel(), data_RGBG[0].ravel(), data_RGBG[1::2].ravel(), data_RGBG[2].ravel()]
    fig, axs = plt.subplots(nrows=4, sharex=True, sharey=True, figsize=(3.3,5), squeeze=True, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0})
    for data, colour, ax in zip(data_KRGB, "kRGB", axs):
        ax.hist(data.ravel(), bins=np.linspace(*xlim, nrbins), color=colour, density=True)
        ax.grid(True)
    for ax in axs[:3]:
        ax.xaxis.set_ticks_position("none")
    axs[0].set_xlim(*xlim)
    axs[3].set_xlabel(xlabel)
    axs[0].set_yscale("log")
    axs[2].set_ylabel(25*" "+"Probability density")
    _saveshow(saveto)
    
