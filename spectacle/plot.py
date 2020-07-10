from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt, patheffects as pe, ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from . import raw
from .wavelength import fluorescent_lines
from .linearity import pearson_r_single
from .general import symmetric_percentiles


# Colour maps for red/green/blue
cmaps = {"R": plt.cm.Reds, "G": plt.cm.Greens, "B": plt.cm.Blues, "G2": plt.cm.Greens,
         "r": plt.cm.Reds, "g": plt.cm.Greens, "b": plt.cm.Blues, "g2": plt.cm.Greens,
         "Rr": plt.cm.Reds_r, "Gr": plt.cm.Greens_r, "Br": plt.cm.Blues_r, "G2r": plt.cm.Greens_r,
         "rr": plt.cm.Reds_r, "gr": plt.cm.Greens_r, "br": plt.cm.Blues_r, "g2r": plt.cm.Greens_r,
         None: plt.cm.viridis}


# Constants for easy iteration
rgb = "rgb"
RGB = "RGB"
rgbg = "rgbg"
rgbg2 = ["r", "g", "b", "g2"]
RGBG2 = ["R", "G", "B", "G2"]


def _convert_to_path(path):
    # Convert to a Path-type object
    try:
        path = Path(path)
    # If `path` cannot be made into a Path, assume it is None and continue
    except TypeError:
        path = None

    return path


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


def plot_fluorescent_spectrum(wavelengths, RGB, saveto=None, ylabel="Digital value (ADU)", xlabel="Wavelength (nm)", **kwargs):
    plt.figure(figsize=(6.6, 3), tight_layout=True)
    _rgbplot(wavelengths, RGB)
    plt.axis("tight")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for line in fluorescent_lines:
        plt.axvline(line, c='0.5', ls="--")
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


def plot_fluorescent_lines(y, lines, lines_fit, saveto=None):
    plt.figure(figsize=(3.3,3))
    _rgbplot(y, lines, func=plt.scatter, s=25)
    p_eff = [pe.Stroke(linewidth=5, foreground='k'), pe.Normal()]
    _rgbplot(y, lines_fit, path_effects=p_eff)
    plt.title("Locations of RGB maxima")
    plt.xlabel("Row along spectrum ($y$)")
    plt.ylabel("Line location ($x$)")
    plt.axis("tight")
    plt.tight_layout()
    _saveshow(saveto)


def RGBG(RGBG, saveto=None, size=13, **kwargs):
    # replace with `show_RGBG`
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


def colorbar(mappable, location="bottom", label=""):
    orientation = "horizontal" if location in ("top", "bottom") else "vertical"
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes(location, size="5%", pad=0.05)
    cbar = fig.colorbar(mappable, cax=cax, orientation=orientation, ticklocation=location)
    cbar.set_label(label)
    return cbar


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


def show_image_RGBG2(data, saveto=None, vmin="auto", vmax="auto", **kwargs):
    # Default vmin and vmax if none are given by the user
    if vmin == "auto" or vmax == "auto":
        if vmin == "auto":
            vmin = symmetric_percentiles(data)[0]
        if vmax == "auto":
            vmax = symmetric_percentiles(data)[1]
    kwargs.update({"vmin": vmin, "vmax": vmax})

    saveto = _convert_to_path(saveto)

    for j, c in enumerate(RGBG2):
        try:
            saveto_c = saveto.parent / (saveto.stem + "_" + c + saveto.suffix)
        except AttributeError:
            saveto_c = None

        show_image(data[j], saveto=saveto_c, colour=c, **kwargs)

    try:
        saveto_RGBG2 = saveto.parent / (saveto.stem + "_RGBG2" + saveto.suffix)
    except AttributeError:
        saveto_RGBG2 = None

    show_RGBG(data, saveto=saveto_RGBG2, **kwargs)


def show_RGBG(data, colour=None, colorbar_label="", saveto=None, **kwargs):
    fig, axs = plt.subplots(ncols=4, sharex=True, sharey=True, figsize=(7,2), squeeze=True, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0})
    for ax, data_c, c in zip(axs, data, RGBG2):
        img = ax.imshow(data_c, cmap=cmaps[c+"r"], **kwargs)
        ax.set_xticks([])
        ax.set_yticks([])
        colorbar_here = colorbar(img)
        if ax is axs[1]:
            colorbar_here.set_label(colorbar_label)
        colorbar_here.locator = ticker.MaxNLocator(nbins=4)
        colorbar_here.update_ticks()
    _saveshow(saveto)


def histogram_RGB(data_RGBG, xmin="auto", xmax="auto", nrbins=500, xlabel="", yscale="linear", saveto=None):
    if xmin == "auto":
        xmin = symmetric_percentiles(data_RGBG)[0]
    if xmax == "auto":
        xmax = symmetric_percentiles(data_RGBG)[1]
    data_KRGB = [data_RGBG.ravel(), data_RGBG[0].ravel(), data_RGBG[1::2].ravel(), data_RGBG[2].ravel()]
    fig, axs = plt.subplots(nrows=4, sharex=True, sharey=True, figsize=(3.3,5), squeeze=True, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0})
    for data, colour, ax in zip(data_KRGB, "kRGB", axs):
        ax.hist(data.ravel(), bins=np.linspace(xmin, xmax, nrbins), color=colour, edgecolor=colour, density=True)
        ax.grid(True)
    for ax in axs[:3]:
        ax.xaxis.set_ticks_position("none")
    axs[0].set_xlim(xmin, xmax)
    axs[3].set_xlabel(xlabel)
    axs[0].set_yscale(yscale)
    axs[2].set_ylabel(25*" "+"Probability density")
    _saveshow(saveto)


def plot_linearity_dng(intensities, means, colours_here, intensities_errors=None, max_value=4095, savefolder=None):
    savefolder = _convert_to_path(savefolder)

    for j in range(4):
        colour_index = colours_here[j]
        colour = "rgbg"[colour_index]
        if colour_index < 3:
            label = colour
        else:
            label = "g2"
        try:
            saveto = savefolder/f"linearity_response_RAW_{label}.pdf"
        except TypeError:
            saveto = None

        mean_dng =  means[:, j]

        r = pearson_r_single(intensities, mean_dng, saturate=max_value*0.95)

        fig, ax2 = plt.subplots(1, 1, figsize=(3.3,2), tight_layout=True)
        ax2.errorbar(intensities, mean_dng, xerr=intensities_errors, fmt="ko", ms=3)
        ax2.set_ylim(0, max_value*1.02)
        ax2.locator_params(axis="y", nbins=5)
        ax2.set_ylabel("RAW value")
        ax2.grid(True)
        ax2.set_title(f"$r = {r:.3f}$")
        _saveshow(saveto)
        print(f"Plotted pixel {j} ({label})")


def plot_linearity_dng_jpg(intensities, means, jmeans, colours_here, intensities_errors=None, max_value=4095, savefolder=None):
    savefolder = _convert_to_path(savefolder)

    for j in range(4):
        colour_index = colours_here[j]
        colour = "rgbg"[colour_index]
        if colour_index < 3:
            i = colour_index
            label = colour
        else:
            i = 1
            label = "g2"
        try:
            saveto = savefolder/f"linearity_response_RAW_JPEG_{label}.pdf"
        except TypeError:
            saveto = None

        mean_dng =  means[:, j]
        mean_jpg = jmeans[:, j, i]

        r_dng  = pearson_r_single(intensities, mean_dng, saturate=max_value*0.95)
        r_jpeg = pearson_r_single(intensities, mean_jpg, saturate=240)
        r_dng_str = "$r_{DNG}"
        r_jpg_str = "$r_{JPEG}"

        fig, ax = plt.subplots(1, 1, figsize=(3.3,2), tight_layout=True)
        ax.errorbar(intensities, mean_jpg, xerr=intensities_errors, fmt=f"{colour}o", ms=3)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(0, 255*1.02)
        ax.set_xticks(np.arange(0,1.2,0.2))
        ax.set_yticks(np.arange(0, 255, 50))
        ax2 = ax.twinx()
        ax2.errorbar(intensities, mean_dng, xerr=intensities_errors, fmt="ko", ms=3)
        ax2.set_ylim(0, max_value*1.02)
        ax2.locator_params(axis="y", nbins=5)
        ax.grid(True, axis="x")
        ax2.grid(True, axis="y")
        jpeglabel = ax.set_ylabel("JPEG value")
        jpeglabel.set_color(colour)
        ax.tick_params(axis="y", colors=colour)
        ax2.set_ylabel("RAW value")
        ax.set_xlabel("Relative incident intensity")
        ax.set_title(f"{r_jpg_str} = {r_jpeg:.3f}$    {r_dng_str} = {r_dng:.3f}$")
        _saveshow(saveto)
        print(f"Plotted pixel {j} ({label})")
