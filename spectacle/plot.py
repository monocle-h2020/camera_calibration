from pathlib import Path
import numpy as np
from matplotlib import pyplot as plt, patheffects as pe, ticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
from . import raw
from .wavelength import fluorescent_lines
from .linearity import pearson_r_single
from .general import symmetric_percentiles

# Default plot settings
plt.rcParams["grid.linestyle"] = "--"

# Colour maps for red/green/blue
cmaps = {"R": plt.cm.Reds, "G": plt.cm.Greens, "B": plt.cm.Blues, "G2": plt.cm.Greens,
         "r": plt.cm.Reds, "g": plt.cm.Greens, "b": plt.cm.Blues, "g2": plt.cm.Greens,
         "Rr": plt.cm.Reds_r, "Gr": plt.cm.Greens_r, "Br": plt.cm.Blues_r, "G2r": plt.cm.Greens_r,
         "rr": plt.cm.Reds_r, "gr": plt.cm.Greens_r, "br": plt.cm.Blues_r, "g2r": plt.cm.Greens_r,
         None: plt.cm.cividis}


# Colour-blind friendly RGB colours, adapted from Okabe-Ito
RGB_OkabeIto = [[213/255, 94/255,  0],
                [0,       158/255, 115/255],
                [0/255,   114/255, 178/255]]

RGBG_OkabeIto = RGB_OkabeIto + [RGB_OkabeIto[1]]  # Just duplicate the G element
kRGB_OkabeIto = ["k", *RGB_OkabeIto]  # black + RGB


# Constants for easy iteration
rgb = "rgb"
RGB = "RGB"
rgbg = "rgbg"
rgby = "rgby"
rgbg2 = ["r", "g", "b", "g2"]
RGBG2 = ["R", "G", "B", "G2"]


# bbox for text
bbox_text = {"boxstyle": "round", "facecolor": "white"}


def _convert_to_path(path):
    # Convert to a Path-type object
    try:
        path = Path(path)
    # If `path` cannot be made into a Path, assume it is None and continue
    except TypeError:
        path = None

    return path


def save_or_show(saveto=None, close=True, bbox_inches="tight", **kwargs):
    """
    If `saveto` is not None, save the figure there; otherwise, show it.
    If `close` is True, then the figure is closed afterwards.
    `bbox_inches`="tight" by default, meaning white space is cut out. Setting it to None prevents this.
    Any additional **kwargs are passed to `plt.savefig` but not to `plt.show`.
    """
    if saveto is None:
        plt.show()
    else:
        plt.savefig(saveto, bbox_inches=bbox_inches, **kwargs)
    if close:
        plt.close()

_saveshow = save_or_show  # Alias for backwards compatibility


def _rgbplot(x, y, func=plt.plot, **kwargs):
    for y_c, c in zip(y, RGB_OkabeIto):
        func(x, y_c, c=c, **kwargs)


def _rgbgplot(x, y, func=plt.plot, **kwargs):
    for y_c, c in zip(y, RGBG_OkabeIto):
        func(x, y_c, c=c, **kwargs)


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
    save_or_show(saveto)


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
    save_or_show(saveto)


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
    save_or_show(saveto)


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
    save_or_show(saveto, transparent=True)


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
    cmap = cmaps[colour+"r"] if colour else plt.cm.cividis
    plt.figure(figsize=(3.3,3*data.shape[0]/data.shape[1]), tight_layout=True)
    img = plt.imshow(data, cmap=cmap, **kwargs)
    plt.xticks([])
    plt.yticks([])
    colorbar_here = colorbar(img)
    colorbar_here.set_label(colorbar_label)
    colorbar_here.locator = ticker.MaxNLocator(nbins=5)
    colorbar_here.update_ticks()
    save_or_show(saveto)


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
    save_or_show(saveto)


def histogram_RGB(data_RGBG, xmin="auto", xmax="auto", nrbins=500, xlabel="", yscale="linear", skip_combined=False, axs=None, saveto=None):
    """
    Make a histogram of RGBG data with panels in black (all combined - optional), red, green (G + G2 combined), and blue.

    Can be done on existing Axes if `axs` are passed.
    """
    # Get upper and lower bounds for the axes
    if xmin == "auto":
        xmin = symmetric_percentiles(data_RGBG)[0]
    if xmax == "auto":
        xmax = symmetric_percentiles(data_RGBG)[1]

    # Unravel the data
    data_KRGB = [data_RGBG.ravel(), data_RGBG[0].ravel(), data_RGBG[1::2].ravel(), data_RGBG[2].ravel()]

    # If no axs were passed, make a new figure
    if axs is None:
        fig, axs = plt.subplots(nrows=3+plot_combined, sharex=True, sharey=True, figsize=(3.3,5), squeeze=True, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0})
        newfig = True
    else:
        newfig = False

    # Loop over the different channels and plot them
    # Starting from plot_combined is an ugly way to skip combined panel if one is not desired
    for data, colour, ax in zip(data_KRGB[skip_combined:], kRGB_OkabeIto[skip_combined:], axs):
        ax.hist(data, bins=np.linspace(xmin, xmax, nrbins), color=colour, edgecolor=colour, density=True)
        ax.grid(ls="--")

    # Plot settings
    for ax in axs[:-1]:
        ax.xaxis.set_ticks_position("none")
    axs[0].set_xlim(xmin, xmax)
    axs[-1].set_xlabel(xlabel)
    axs[0].set_yscale(yscale)

    # Only include a ylabel if a new figure was made
    if newfig:
        axs[2].set_ylabel(25*" "+"Probability density")

        # Save or show the result
        save_or_show(saveto)


def plot_linearity_dng(intensities, means, colours_here, intensities_errors=None, max_value=4095, savefolder=None):
    savefolder = _convert_to_path(savefolder)

    for j in range(4):
        colour_index = colours_here[j]
        colour = rgbg[colour_index]
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
        save_or_show(saveto)
        print(f"Plotted pixel {j} ({label})")


def plot_linearity_dng_jpg(intensities, means, jmeans, colours_here, intensities_errors=None, max_value=4095, savefolder=None):
    savefolder = _convert_to_path(savefolder)

    for j in range(4):
        colour_index = colours_here[j]
        colour = rgbg[colour_index]
        if colour_index < 3:
            i = colour_index
            label = colour
        else:
            i = 1,
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
        save_or_show(saveto)
        print(f"Plotted pixel {j} ({label})")


def plot_covariance_matrix(matrix, label="Covariance", title="", nr_bins=None, majorticks=None, minorticks=None, ticklabels=None, saveto=None, **kwargs):
    """
    Plot a covariance (or correlation) matrix.
    """
    # Get a segmented colourmap if wanted
    cmap = plt.cm.get_cmap("cividis", nr_bins)

    # Make a figure
    plt.figure(figsize=(6,6))

    # Plot the data
    plt.imshow(matrix, cmap=cmap, origin="lower", extent=(0, *matrix.shape, 0), **kwargs)

    # Labels on the axes and colour bar
    plt.colorbar(label=label)
    plt.title(title)

    ax = plt.gca()
    for axis in [ax.xaxis, ax.yaxis]:
        if majorticks is not None:
            axis.set_ticks(majorticks)
        if minorticks is not None:
            axis.set_ticks(minorticks, minor=True)
        axis.set_tick_params(which="major", labelleft=False, labelbottom=False)
        axis.set_tick_params(which="minor", left=False, bottom=False)

    # Set the ticklabels normally on x and in reverse on y
    if ticklabels is not None:
        ax.set_xticklabels(ticklabels, minor=True)
        ax.set_yticklabels(ticklabels[::-1], minor=True)

    # Save/show result
    save_or_show(saveto)
