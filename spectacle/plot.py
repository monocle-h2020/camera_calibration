from pathlib import Path
from functools import partial
import numpy as np
from matplotlib import pyplot as plt, patheffects as pe, ticker, colors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from cmcrameri import cm

from . import raw
from .wavelength import fluorescent_lines
from .linearity import pearson_r_single
from .general import symmetric_percentiles


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
RGBG2_latex = [f"${c}$" for c in [*RGB, "G_2"]]
RGB_latex = RGBG2_latex[:3]

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
        plt.savefig(saveto, bbox_inches="tight", **kwargs)
    if close:
        plt.close()


def _rgbplot(x, y, func=plt.plot, **kwargs):
    for y_c, c in zip(y, RGB_OkabeIto):
        func(x, y_c, color=c, **kwargs)


def _rgbgplot(x, y, func=plt.plot, **kwargs):
    for y_c, c in zip(y, RGBG_OkabeIto):
        func(x, y_c, color=c, **kwargs)


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


def histogram_RGB(data_RGBG, xmin="auto", xmax="auto", nrbins=500, xlabel="", yscale="linear", axs=None, saveto=None):
    """
    Make a histogram of RGBG data with panels in black (all combined), red, green
    (G + G2 combined), and blue.

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
        fig, axs = plt.subplots(nrows=4, sharex=True, sharey=True, figsize=(3.3,5), squeeze=True, tight_layout=True, gridspec_kw={"wspace":0, "hspace":0})
        newfig = True
    else:
        newfig = False

    # Loop over the different channels and plot them
    for data, colour, ax in zip(data_KRGB, kRGB_OkabeIto, axs):
        ax.hist(data, bins=np.linspace(xmin, xmax, nrbins), color=colour, edgecolor=colour, density=True)
        ax.grid(ls="--")

    # Plot settings
    for ax in axs[:3]:
        ax.xaxis.set_ticks_position("none")
    axs[0].set_xlim(xmin, xmax)
    axs[-1].set_xlabel(xlabel)
    axs[0].set_yscale(yscale)

    # Only include a ylabel if a new figure was made
    if newfig:
        axs[2].set_ylabel(25*" "+"Probability density")

        # Save or show the result
        _saveshow(saveto)


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
        _saveshow(saveto)
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
        _saveshow(saveto)
        print(f"Plotted pixel {j} ({label})")


def get_tick_locations_from_slices(slices):
    """
    From a list of slices, get the corresponding major and minor ticks.
    Major ticks are located at the start of each slice and the end of the final slice.
    Minor ticks are located halfway between the start and end of each slice.
    """
    slices = np.ravel(slices)
    ticks_major = [s.start for s in slices] + [slices[-1].stop]
    ticks_minor = [(s.start + s.stop) / 2 for s in slices]

    return ticks_major, ticks_minor


def plot_covariance_matrix(matrix, label="Covariance", title="", nr_bins=None, majorticks=None, minorticks=None, ticklabels=None, saveto=None, cmap="cividis", **kwargs):
    """
    Plot a covariance (or correlation) matrix.
    """
    # Get a segmented colourmap if desired
    if isinstance(cmap, str):  # If a matplotlib cmap label was given
        cmap = plt.cm.get_cmap("cividis")

    if nr_bins is not None:
        cmap = cmap._resample(nr_bins)

    # Make a figure
    plt.figure(figsize=(6,6))

    # Plot the data
    plt.imshow(matrix, cmap=cmap, origin="lower", extent=(0, matrix.shape[0], 0, matrix.shape[1]), **kwargs)

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

    # Set the ticklabels
    if ticklabels is not None:
        ax.set_xticklabels(ticklabels, minor=True)
        ax.set_yticklabels(ticklabels, minor=True)

    # Save/show result
    _saveshow(saveto)


# Variant of plot_covariance_matrix with pre-filled kwargs for correlation matrices
plot_correlation_matrix = partial(plot_covariance_matrix, label="Correlation", cmap=cm.lisbon, vmin=-1, vmax=1)


def plot_correlation_matrix_diagonal(correlation, slices, wavelengths, xlim=(390, 700), offset=1, xlabel="Wavelength/nm", ax=None, saveto=None):
    """
    Plot diagonal lines in correlation matrices. By default this plots the diagonal with offset 1,
    i.e. the correlation between elements (n, n+1).
    A full correlation matrix should be provided, with appropriate slices corresponding to the different bands.

    If an Axes object is given for `ax`, plot into that. Otherwise, make a new plot.
    """
    # Check if we are making a new plot or plotting into an existing object
    if ax is None:
        plt.figure(figsize=(5,3))
        ax = plt.gca()
        newplot = True
    else:
        newplot = False

    # Get the correlation between subsequent elements for each band
    correlation_with_next = np.array([np.diagonal(correlation[s,s], offset=1) for s in slices])

    # Plot the correlation diagonals in a single pane
    _rgbplot(wavelengths[:-offset], correlation_with_next, func=ax.plot)

    # Figure settings
    ax.set_xlim(xlim)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Correlation with\nnext element")
    ax.grid(ls="--")

    # Optional figure settings if this is a new (stand-alone) plot
    if newplot:
        ax.set_xlabel(xlabel)
        _saveshow(saveto)


def plot_correlation_matrix_diagonal_multi(correlation, slices, wavelengths, xlabel="Wavelength/nm", saveto=None, **kwargs):
    """
    Plot diagonal lines in correlation matrices. By default this plots the diagonal with offset 1,
    i.e. the correlation between elements (n, n+1).
    A full correlation matrix should be provided, with appropriate slices corresponding to the different bands.

    This function plots the correlations in each data set within the correlation matrix in a separate panel.
    **kwargs are passed to plot_correlation_matrix_diagonal.
    """
    # Create a figure to hold the panels
    nr_panels = len(slices)
    fig, axs = plt.subplots(nrows=nr_panels, sharex=True, sharey=True, figsize=(5, 3*nr_panels))

    # Loop over the data and plot each
    for ax, slice_list, wavelength_list in zip(axs, slices, wavelengths):
        plot_correlation_matrix_diagonal(correlation, slice_list, wavelength_list, ax=ax, **kwargs)

    # Figure settings
    axs[-1].set_xlabel(xlabel)

    # Save or show the result
    _saveshow(saveto)


def plot_correlation_matrix_diagonal_RGBG2(correlation, slices, wavelengths, xlim=(390, 700), xlabel="Wavelength/nm", axs=None, saveto=None):
    """
    Plot correlations between equal wavelengths in the RGBG2 bands.
    For each band, it plots the correlations with the other three bands.
    A full correlation matrix should be provided, with appropriate slices corresponding to the different bands.

    If an iterable of 4 Axes objects is given for axs, use those. Otherwise, make a new plot.
    """
    # Check if we are making a new plot or plotting into an existing object
    if axs is None:
        fig, axs = plt.subplots(nrows=4, figsize=(5,6), sharex=True, sharey=True)
        newplot = True
    else:
        newplot = False

    # Get the correlation between corresponding elements of the different bands
    correlation_between_bands = np.array([[np.diagonal(correlation[s1, s2]) for s2 in slices] for s1 in slices])

    # Plot the correlation diagonals in each band in the different panes
    for ax, corr in zip(axs, correlation_between_bands):
        _rgbgplot(wavelengths, corr, func=ax.plot)

    # Figure settings
    for ax, label in zip(axs, RGBG2_latex):
        ax.set_xlim(xlim)
        ax.set_ylim(-1, 1)
        ax.set_ylabel(f"Correlation\nwith {label}")
        ax.set_yticks(np.arange(-1, 1.01, 0.5))
        ax.grid(ls="--")

    # Optional figure settings if this is a new (stand-alone) plot
    if newplot:
        ax.set_xlabel(xlabel)
        _saveshow(saveto)


def plot_correlation_matrix_diagonal_RGBG2_multi(correlation, slices, wavelengths, xlabel="Wavelength/nm", saveto=None, **kwargs):
    """
    Plot correlations between equal wavelengths in the RGBG2 bands.
    For each band, it plots the correlations with the other three bands.
    A full correlation matrix should be provided, with appropriate slices corresponding to the different bands.

    This function plots the correlations in each data set within the correlation matrix in a separate column of panels.
    **kwargs are passed to plot_correlation_matrix_diagonal_RGBG2.
    """
    # Create a figure to hold the panels
    nr_columns = len(slices)
    fig, axs = plt.subplots(nrows=4, ncols=nr_columns, sharex=True, sharey=True, figsize=(4*nr_columns, 8))
    axs = axs.T

    # Loop over the data and plot each
    for j, (ax_col, slice_list, wavelength_list) in enumerate(zip(axs, slices, wavelengths)):
        plot_correlation_matrix_diagonal_RGBG2(correlation, slice_list, wavelength_list, axs=ax_col, **kwargs)

        # Figure settings
        ax_col[0].set_title(f"Data set {j}")
        ax_col[-1].set_xlabel(xlabel)

    # Remove y-axis labels from columns that are not on the left side
    for ax in np.ravel(axs[1:]):
        ax.tick_params(axis="y", left=False, labelleft=False)

    # Add y-axis labels to the right-most column
    for ax in np.ravel(axs[-1]):
        ax.yaxis.set_label_position("right")
        ax.tick_params(axis="y", right=True, labelright=True)

    # Save or show the result
    _saveshow(saveto)
