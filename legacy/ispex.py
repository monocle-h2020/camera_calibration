"""
Olivier Burggraaff
Leiden University
Bacheloronderzoek: Smartphone Spectrometry

Functions
"""

import numpy as np
from matplotlib import pyplot as plt
import PIL  # Python Image Library - also a very good band from the 80s
import sys
import os
import time
from scipy import stats as st  # we use this for st.nanmedian()
# this is deprecated and should be replaced with numpy.nanmedian()

# the start of the slit is always between 18% and 25% of the image size
slitstartpct = np.array([0.18, 0.25])
# the end of the slit is always between 51% and 58% of the image size
slitendpct = np.array([0.51, 0.58])
#range for finding the middle of the slit - within 6% of the average
#of start and end
midpct = 0.06

pct = 0.04  # the percentage of lines after the slit start, before the
# slit end, and around the slit middle that we ignore to compensate
# for inaccuracies

#ratio between thick and thin slits (both in mm)
ratio = 0.4/0.17

#the RGB channels - this array is used for easy plotting
colours = ['red', 'green', 'blue']

#the four important peaks in the fluorescent light spectrum (nm)
TLpeaks = np.array([436.6, 487.7, 544.45, 611.6])

#the wavelengths we will be looking at
wavelengthrange = (380, 700)
#how wide the resample range is (e.g. for 400 nm, one gets the average
#of the values between 400-resamplewidth/2 and 400+resamplewidth/2 nm
resamplewidth = 5.0 #nm
#the centres of the wavelength bins we will be using
bincentres = np.arange(wavelengthrange[0], wavelengthrange[1], 1.0)

T_sun = 5778 #K, temperature of the sun
h = 6.62606957e-34 #J s, Planck's constant
c = 299792458.0 #m s^-1, speed of light
kB = 1.3806488e-23 #J K^-1, Boltzmann's constant


def bb(wvl, T):
    """
    Calculates the normalised Black-Body radiation spectrum

    wvl: wavelengths, in nm
    T: temperature, in K
    """
    wvl2 = wvl*10.0**-9.0  # conversion to meters
    intensity = wvl2**(-5.0) * 1.0/(np.exp(h*c/(wvl2*kB*T)))
    # we can ignore the other constants because we will normalise anyway
    return intensity/np.amax(intensity)


def get_filelist(directory, imgpre):
    """
    Returns a sorted list of images in a directory starting with a
    certain string imgpre

    directory: directory to search in
    imgpre: what the image name should start with (e.g. IMG, sun, tl)
    """
    filelist = []
    for file in os.listdir(directory):
        if file.startswith(imgpre) and file.endswith(".JPG"):
            filelist.append(directory+file)

    return np.sort(filelist)


def importimage(filename, bulk=False):
    """
    Import the image "filename"

    filename: the name of the file to be imported
    bulk: if True, no diagnostics are printed
    """
    try:
        img = PIL.Image.open(filename)
    except IOError:
        sys.tracebacklimit = 0  # prevent traceback
        raise IOError("Could not open file: "+filename)
    if not bulk:
        print "\n@@@ imported image:", filename, "@@@"
    return img


def importwavelength(rowstart, rowend, imgsize, slitpos, filename="fitparameters.txt", bulk=False, makeplot=False):
    """
    Imports the wavelength fit parameters
    Calculates the wavelengths for all pixels between rowstart and
    rowend, for the last half of the pixels (horizontally)
    """
    if not bulk:
        print "\n@@@ importing wavelengths @@@"
    is0 = int(imgsize[0]/2)
    try:
        param = np.loadtxt(filename)
    except IOError:
        if not filename == "fitparameters.txt":
            imgstr = str(imgsize[0])+"_"+str(imgsize[1])
            try:
                param = np.loadtxt("fitparameters"+imgstr+".txt")
                filename = "fitparameters"+imgstr+".txt"
            except IOError:
                sys.tracebacklimit = 0  # prevent traceback
                raise IOError("Could not open file: "+filename+" nor fitparameters"+imgstr+".txt")
        else:
            sys.tracebacklimit = 0  # prevent traceback
            raise IOError("Could not open file: fitparameters.txt")
    if not bulk:
        print "imported fit parameters from file", filename

    # we can assume the spectrum to be between rows ss and se
    ss = int(slitstartpct[0] * imgsize[1])
    se = int(slitendpct[1] * imgsize[1])

    if (ss != param[0,0] or se != param[-1,0]):
        sys.tracebacklimit = 0 #prevent traceback
        raise ValueError(filename+" size does not correspond to current image")

    x = np.tile(np.nan, (rowend-rowstart+1, is0))
    wvl = np.tile(np.nan, (rowend-rowstart+1, is0))
    for j in range(rowend-rowstart+1):
        x[j] = np.arange(is0, imgsize[0]) - slitpos[j]

    # we remove the entries of param that are not related to rows we've determined
    # the spectrum-to-inspect to be in
    param = np.delete(param, range(rowstart-ss), axis=0)
    param = np.delete(param, range(rowend-rowstart+1, se-rowstart+1), axis=0)
    for j in range(rowend-rowstart+1):
        wvl[j] = np.polyval(param[j, 1:], x[j])

    if makeplot:
        plt.figure(figsize=(20, 10))
        plt.plot(x[0], wvl[0], c='r', label='row '+str(rowstart), lw=2)
        m = (rowend+rowstart)/2 - rowstart
        plt.plot(x[m], wvl[m], c='g', label='row '+str(m+rowstart), lw=2)
        plt.plot(x[-1], wvl[-1], c='b', label='row '+str(rowend), lw=2)
        plt.ylim(380, 700)
        plt.legend(loc='lower right')
        plt.title("Wavelength fit: wavelength as function of pixel column number for first, middle and last rows in image")
        plt.xlabel("Column number (from slit)")
        plt.ylabel("Wavelength (nm)")
        if filename == "fitparameters.txt":
            plt.savefig("wavelengthfit.png")
        else:
            folder = os.path.split(filename)[0]
            plt.savefig(folder+"\\wavelengthfit.png")
        plt.close()

    if not bulk:
        print "@@@ finished importing wavelengths @@@"
    return wvl


def importfilters(filename="filtercurves.txt", nr=False, testing=False):
    """
    Imports the filter curves

    Returns bincentres, filter curves, number of files used to
    calculate filter curve for each bin (if nr=True)
    """
    try:
        bincentres = np.loadtxt(filename, usecols=[0])
    except IOError:
        if not filename == "filtercurves.txt":
            try:
                bincentres = np.loadtxt("filtercurves.txt", usecols=[0])
                filename = "filtercurves.txt"
            except IOError:
                sys.tracebacklimit = 0  # prevent traceback
                raise IOError("Could not open file: "+filename+" nor filtercurves.txt")
        else:
            sys.tracebacklimit = 0  # prevent traceback
            raise IOError("Could not open file: filtercurves.txt")
    if testing:
        print "\n@@@ imported bincentres from file", filename, "@@@"

    try:
        filtercurves = np.loadtxt(filename, usecols=[1, 2, 3])
    except IndexError:
        print "$$$ could not import filtercurves - returning nan $$$"
        filtercurves = np.tile(np.nan, (len(bincentres), 3))

    if nr:
        try:
            nrs = np.loadtxt(filename, usecols=[4, 5, 6])
        except IndexError:
            print "$$$ could not import nrs - returning nan $$$"
            nrs = np.tile(np.nan, (len(bincentres), 3))
        if testing:
            print "@@@ returned bincentres, filtercurves, nrs @@@"
        return bincentres, filtercurves, nrs

    if testing:
        print "@@@ returned bincentres, filtercurves @@@"
    return bincentres, filtercurves


def importgamma(filename="gamma.txt", testing=False):
    try:
        gamma = np.loadtxt(filename)
    except IOError:
        if not filename == "gamma.txt":
            try:
                gamma = np.loadtxt("gamma.txt")
                filename = "gamma.txt"
            except IOError:
                sys.tracebacklimit = 0  # prevent traceback
                raise IOError("Could not open file: "+filename+" nor gamma.txt")
        else:
            sys.tracebacklimit = 0  # prevent traceback
            raise IOError("Could not open file: gamma.txt")

    if testing:
        print "@@@ imported gamma of", gamma, "from file", filename, "@@@"

    return gamma[()]


def importwhitebalance(filename="whitebalance.txt", testing=False):
    try:
        wb = np.loadtxt(filename)
    except IOError:
        if not filename == "whitebalance.txt":
            try:
                wb = np.loadtxt("whitebalance.txt")
                filename = "whitebalance.txt"
            except IOError:
                sys.tracebacklimit = 0  # prevent traceback
                raise IOError("Could not open file: "+filename+" nor whitebalance.txt")
        else:
            sys.tracebacklimit = 0  # prevent traceback
            raise IOError("Could not open file: whitebalance.txt")

    if testing:
        print "@@@ imported whitebalance of", wb, "from file", filename, "@@@"

    return wb


def findslit(img, bulk=True, safe=True):
    """
    Given an img object, determine the location of the slit

    img: image object as given by importimage()
    bulk: if True, no diagnostics are printed and no plots made
    safe: if True, rowstart and rowend are moved to 'safe' areas and
    slitpos is trimmed accordingly
    """
    pixels = img.load()

    if not bulk:
        print "\n@@@ finding slit @@@"
        plt.figure(figsize=(20, 10))
        plt.xlabel("Row")
        plt.ylabel("r/g/b sum over row")
        plt.title("r/g/b sum over rows, with slit indicated")
        plt.xlim(0, img.size[1])

    avg = np.max([5, int(img.size[1]/200)])  # how many rows we average over

    slitparams = np.array([0, 0, 0], dtype=int)  # slitstart, slitmiddle, slitend
    starts = np.array([int(img.size[1] * slitstartpct[0]), 0, int(img.size[1] * slitendpct[0])], dtype=int)
    ends = np.array([int(img.size[1] * slitstartpct[1]), 0, int(img.size[1] * slitendpct[1])], dtype=int)
    sizes = ends - starts
    # the functions we use
    funcs = [np.nanargmax, np.nanargmax, np.nanargmin]

    whatwedo = ['START', 'MIDDLE', 'END']

    for z in [0, 2, 1]:
        if not bulk:
            print "FINDING SLIT", whatwedo[z]

        if z == 1:  # this is such an ugly way to handle this
            aver = 0.5*(slitparams[0] + slitparams[2])
            starts[1] = int(aver*(1.0-midpct))
            ends[1] = int(aver*(1.0+midpct))
            sizes[1] = ends[1] - starts[1]

        # this will contain the summed r,g,b values per row
        rgb = np.zeros((sizes[z], 3))
        # this will contain the r,g,b values of row i before they are
        # summed up to be put in rgb
        rgb2 = np.zeros((img.size[0], 3))

        for j in range(starts[z], ends[z]):  # row
            for i in range(img.size[0]):
                rgb2[i] = pixels[i, j]

            rgb[j-starts[z]] = np.sum(rgb2, axis=0)

        rgbdif = np.diff(rgb, axis=0)
        rgbdifavg = np.zeros((sizes[z], 3))
        for j in range(avg, sizes[z]-avg):
            rgbdifavg[j] = np.mean(rgbdif[j-avg:j+avg], axis=0)

        slitparams[z] = int(st.nanmedian(funcs[z](rgbdifavg, axis=0)) + starts[z])

        if not bulk:
            xrng = range(starts[z], ends[z])
            for c in [0, 1, 2]:
                plt.scatter(xrng, rgb[:, c], color=colours[c], s=2)

            plt.axvline(slitparams[z], c='y')

    if not bulk:
        plt.savefig(os.path.splitext(img.filename)[0]+"_slit.png")
        plt.close()

    # last but not least we wish to find the position of the slit
    # horizontally
    if not bulk:
        print "FINDING SLIT HORIZONTALLY"
    halfimgsize = img.size[0]/2
    rgb = np.zeros((halfimgsize, 3))
    slitrange = np.arange(slitparams[0], slitparams[2])
    center = np.tile(np.nan, (slitparams[2] - slitparams[0]))
    for j in slitrange:  # row
        for i in range(halfimgsize):
            rgb[i] = pixels[i, j]

        center[j-slitparams[0]] = np.mean(np.where(rgb == np.amax(rgb, axis=0))[0])
        # we don't use argmax here because there can be multiple pixels
        # with the max value

    f = np.polyfit(slitrange, center, 1)
    centerfit = np.polyval(f, slitrange)

    if safe:
        slitstart = int(slitparams[0] + pct*(slitparams[2] - slitparams[0]))
        slitend = int(slitparams[2] - pct*(slitparams[2] - slitparams[0]))
        slitpos = centerfit[slitstart-slitparams[2]:slitend-slitparams[2]+1]

        return slitstart, slitparams[1], slitend, slitpos

    return slitparams, centerfit


def rgbbins(bincentres):
    """
    Gives the bins where we have determined (prior knowledge) the
    r,g,b pixels values to be relevant

    This can probably be rewritten to a much simpler form
    """
    edges = np.array([[580, 680], [480, 600], [400, 510]])
    rgbbins = [[], [], []]
    for c in [0, 1, 2]:
        low = np.where(bincentres > edges[c][0])[0][0]
        high = np.where(bincentres <= edges[c][1])[0][-1]
        rgbbins[c] = range(low, high+1)

    return rgbbins


def plotbinlines(rgbbins1, bincentres, crange=[0, 1, 2]):
    """
    Plots a shaded area of rgbbins for each of the colours in crange
    """
    for c in crange:
        plt.axvspan(bincentres[rgbbins1[c][0][0]], bincentres[rgbbins1[c][0][-1]], facecolor=colours[c], alpha=0.15)


def resample(rowstart, rowmid, rowend, rgb, bincentres, wavelengths, testing=False, bulk=False):
    """
    Resamples thin and thick slit spectra (collapsing in wavelength)
    """
    if testing:
        a = time.clock()
        print "\n@@@ resampling @@@"

    thinbins = np.tile(np.nan, (len(bincentres), 3))
    thickbins = np.copy(thinbins)

    mid1 = int(rowmid*(1.0-pct)) - rowstart + 1
    mid2 = int(rowmid*(1.0+pct)) - rowstart

    for l in range(len(bincentres)):
        thins = np.where(np.abs(wavelengths[:mid1] - bincentres[l]) <= resamplewidth/2.0)
        thicks = np.where(np.abs(wavelengths[mid2:] - bincentres[l]) <= resamplewidth/2.0)
        for c in [0,1,2]:
            thinbins[l,c] = np.nanmean(rgb[c,:mid1][thins])
            thickbins[l,c] = np.nanmean(rgb[c,mid2:][thicks])

    if testing:
        print "time elapsed:", time.clock() - a
        print "@@@ done @@@"
    return thinbins, thickbins