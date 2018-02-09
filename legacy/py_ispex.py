"""
Olivier Burggraaff
Leiden University
Bacheloronderzoek: Smartphone Spectrometry

Functions
"""

import numpy as np
from matplotlib import pyplot as plt
import PIL
import sys
import os
import time
import warnings

ssbp = 0.18 #the slit never starts at less than 18% of the image size
ssbp2 = 0.25 #the slit never starts at more than 25% of the image size
ssep = 0.51 #the slit never ends at less than 51% of the image size
ssep2 = 0.58 #the slit never ends at more than 58% of the image size
#range for finding the middle will be determined after start and end are found:
#4% on either side of the average(start, end)
mid = 0.06

pct = 0.04 #the percentage of lines after the slit start, before the slit end, and
#around the slit middle that we ignore to compensate for inaccuracies

colours = ['red', 'green', 'blue']

TLpeaks = np.array([436.6,487.7,544.45, 611.6])

T_sun = 5778 #K, temperature of the sun

h = 6.62606957e-34 #J s
c = 299792458.0 #m s^-1
kB = 1.3806488e-23 #J K^-1
def bb(T, wvl): #T in Kelvin, wvl in nm
    """
    Calculates the Black-Body radiation spectrum at given wavelengths
    """
    wvl2 = wvl * 10.0**-9.0
    intensity = wvl2**(-5.0) * 1.0/(np.exp(h*c/(wvl2*kB*T)))
    #we can ignore the first constants because we will normalise anyway
    return intensity/np.amax(intensity)

def get_filelist(directory, imgtype):
	'''
	Returns a sorted list of all files in given directory of a given type
	'''
	filelist = []
	for file in os.listdir(directory):
		if file.startswith(imgtype) and file.endswith(".JPG"):
			filelist.append(directory+file)

	return np.sort(filelist)

def plotall(rgb, wavelengths, rowstart, rowmid):
    rowmid1 = int(rowmid*(1.0-pct))
    rowmid2 = int(rowmid*(1.0+pct))
    plt.figure(figsize=(20,10))
    plt.xlabel("wavelength (nm)")
    plt.ylabel("r/g/b value")
    plt.title("rgb values of all rows")
    plt.ylim(-1, 260)
    plt.axhline(255, c='k')
    plt.xlim(350, 700)
    for i in [0,1,2]:
        for j in range(rowmid1-rowstart):
            plt.plot(wavelengths[j], rgb[i,j], color=colours[i], lw=1)
        for j in range(rowmid2-rowstart, wavelengths.shape[0]):
            plt.plot(wavelengths[j], rgb[i,j], color=colours[i], lw=3)

def exifinfo(exif):
    """
    Prints a list of all exif tags with their respective value.
    """
    print "\n"
    for k in exif:
        print k, ":", exif.get(k), "\n"

def horilines():
    plt.axhline(487.7, color='black')
    plt.axhline(np.average((542.4, 546.5)), color='black')
    plt.axhline(611.6, color='black')
    plt.axhline(436.6, color='black')

def vertlines():
    plt.axvline(487.7, color='black')
    plt.axvline(np.average((542.4, 546.5)), color='black')
    plt.axvline(611.6, color='black')
    plt.axvline(436.6, color='black')

def importimage(filename, bulk=False):
    """
    Import the image "filename"
    Raises an error if it doesn't work
    """
    try:
        img = PIL.Image.open(filename)
    except IOError:
        sys.tracebacklimit=0 #prevent traceback
        raise IOError("Could not open file: "+filename)
	if not bulk:
		print "\n@@@ imported image:", filename, "@@@"
    return img

def importwavelength(rowstart, rowend, imgsize, slitpos, filename="fitparameters.txt", testing=False, bulk=False):
    """
    Imports the wavelength fit parameters
    Calculates the wavelengths for all pixels between rowstart and rowend, for the last half of the pixels (horizontally)
    """
    a = time.clock()
    is0 = int(imgsize[0]/2)
    if not bulk:
		print "\n@@@ importing fit parameters @@@"
    try:
        param = np.loadtxt(filename)
    except IOError:
        if not filename == "fitparameters.txt":
            try:
                param = np.loadtxt("fitparameters.txt")
                filename = "fitparameters.txt"
            except IOError:
                sys.tracebacklimit=0 #prevent traceback
                raise IOError("Could not open file: "+filename+" nor fitparameters.txt")
        else:
            sys.tracebacklimit=0 #prevent traceback
            raise IOError("Could not open file: fitparameters.txt")
    if not testing and not bulk:
		print "imported fit parameters from file", filename

    ss = int(ssbp*imgsize[1]) #we can assume the spectrum to be between rows
    se = int(ssep2*imgsize[1]) #ss and se

    if (ss != param[0,0] or se != param[-1,0]):
        sys.tracebacklimit=0 #prevent traceback
        raise ValueError(filename+" size does not correspond to current image")

    x = np.tile(np.nan, (rowend-rowstart+1, is0))
    wvl = np.tile(np.nan, (rowend-rowstart+1, is0))
    for j in range(rowend-rowstart+1):
        x[j] = np.arange(is0,imgsize[0]) - slitpos[j]
    #we remove the entries of param that are not related to rows we've determined
    #the spectrum-to-inspect to be in
    param = np.delete(param, range(rowstart-ss), axis=0)
    param = np.delete(param, range(rowend-rowstart+1, se-rowstart+1), axis=0)
    for j in range(rowend-rowstart+1):
        wvl[j] = param[j,1] * x[j]**3.0 + param[j,2] * x[j]**2.0 + param[j,3] * x[j] + param[j,4]
    #print wvl
	if not bulk:
		print "time elapsed:", time.clock()-a, "seconds"
		print "@@@ finished @@@"
    return wvl

def findslit2(pixels, imgsize, imagename, testing=False, bulk=False):
    """
    Given an img.load() object, determine the location of the slit

    Determining the slit: For its horizontal edges, we add the r,g,b values for every
    row together (i.e. r_tot[i] = sum(r[i,:]) ) in pseudocode
    We then check in which rows this value is very large, and decide that those
    constitute the slit
    """
    if not bulk:
		begintime = time.clock()
		print "\n@@@ finding slit @@@"
    avg = np.max([5,int(imgsize[1]/200)]) #how many rows we average over
    
    #first we find the start of the slit
    if testing:
		print "\n@@@ FINDING SLIT START @@@"
    startstart = int(imgsize[1] * ssbp)
    startend = int(imgsize[1] * ssbp2)
    size1 = startend - startstart
    rgb = np.zeros((size1, 3)) #this will contain the summed r,g,b values per row
    #0 = r, 1 = g, 2 = b
    rgb2 = np.zeros((imgsize[0], 3)) #this will contain the r,g,b values of
    #row i before they are summed up to be put in rgb
    for j in range(startstart, startend): #row
        for i in range(imgsize[0]):
            rgb2[i] = pixels[i,j]
        rgb[j-startstart] = np.sum(rgb2, axis=0)
    rgbdif = np.zeros((size1,3))
    rgbdifavg = np.zeros((size1,3))
    for j in range(1,size1):
        rgbdif[j] = rgb[j]-rgb[j-1]
    for j in range(avg,size1-avg):
        rgbdifavg[j] = np.mean(rgbdif[j-avg:j+avg], axis=0)
    slitstart = int(np.mean(rgbdifavg.argmax(axis=0)) + startstart)
    if testing:
		plt.figure(figsize=(20,10))
		plt.xlabel("Row")
		plt.ylabel("r/g/b sum over row")
		plt.title("r/g/b sum over rows, with slit indicated")
		plt.scatter(range(startstart,startend), rgb[:,0], color='red',s=2)
		plt.scatter(range(startstart,startend), rgb[:,1], color='green',s=2)
		plt.scatter(range(startstart,startend), rgb[:,2], color='blue',s=2)
		plt.xlim(0, imgsize[1])
		plt.axvline(slitstart, c='y')
    
    #now we find the end of the slit the exact same way, except we put it at
    #the minimum of rgbdifavg, not the maximum
    if testing:
		print "@@@ FINDING SLIT END @@@"
    endstart = int(imgsize[1] * ssep)
    endend = int(imgsize[1] * ssep2)
    size2 = endend - endstart
    rgb = np.zeros((size2,3))
    rgb2 = np.zeros((imgsize[0], 3))
    for j in range(endstart, endend):
        for i in range(imgsize[0]):
            rgb2[i] = pixels[i,j]
        rgb[j-endstart] = np.sum(rgb2, axis=0)
    rgbdif = np.zeros((size2,3))
    rgbdifavg = np.zeros((size2,3))
    for j in range(1,size2):
        rgbdif[j] = rgb[j] - rgb[j-1]
    for j in range(avg, size2-avg):
        rgbdifavg[j] = np.mean(rgbdif[j-avg:j+avg], axis=0)
    slitend = int(np.mean(rgbdifavg.argmin(axis=0)) + endstart)
    if testing:
		plt.scatter(range(endstart,endend), rgb[:,0], color='red',s=2)
		plt.scatter(range(endstart,endend), rgb[:,1], color='green',s=2)
		plt.scatter(range(endstart,endend), rgb[:,2], color='blue',s=2)
		plt.axvline(slitend, c='y')
    
    #now that we have the slitstart and slitend, we can find the slitmiddle
    if testing:
		print "@@@ FINDING SLIT MIDDLE @@@"
    startendavg = 0.5*(slitstart+slitend)
    middlestart = int(startendavg*(1.0-mid))
    middleend = int(startendavg*(1.0+mid))
    size3 = middleend - middlestart
    rgb = np.zeros((size3,3))
    rgb2 = np.zeros((imgsize[0],3))
    for j in range(middlestart, middleend):
        for i in range(imgsize[0]):
            rgb2[i] = pixels[i,j]
        rgb[j-middlestart] = np.sum(rgb2, axis=0)
    rgbdif = np.zeros((size3,3))
    rgbdifavg = np.zeros((size3,3))
    for j in range(1,size3):
        rgbdif[j] = rgb[j] - rgb[j-1]
    for j in range(avg, size3-avg):
        rgbdifavg[j] = np.mean(rgbdif[j-avg:j+avg], axis=0)
    slitmiddle = int(np.mean(rgbdifavg.argmax(axis=0)) + middlestart)
    if testing:
		plt.scatter(range(middlestart,middleend), rgb[:,0], color='red',s=2)
		plt.scatter(range(middlestart,middleend), rgb[:,1], color='green',s=2)
		plt.scatter(range(middlestart,middleend), rgb[:,2], color='blue',s=2)
		plt.axvline(slitmiddle, c='y')
		plt.savefig(imagename+"_slit.png")
		plt.close()
    
    #last but not least we wish to find the position of the slit horizontally
    #and whether or not is is tilted
    #print "@@@ FINDING SLIT HORIZONTALLY @@@"
    halfimgsize = int(imgsize[0]/2.0)
    rgb = np.zeros((halfimgsize, 3))
    center = np.tile(np.nan, (slitend-slitstart))
    for j in range(slitstart, slitend): #row
        for i in range(halfimgsize):
            rgb[i] = pixels[i,j]
        center[j-slitstart] = np.mean(np.where(rgb == np.amax(rgb, axis=0))[0])
    f = np.polyfit(range(slitstart,slitend), center,1)
    centerfit = f[0] * np.arange(slitstart, slitend) + f[1]
    #print centerfit[0], centerfit[-1]
    #cf = np.median(center)
    #centerfit = np.tile(cf, (slitend-slitstart))
    
    if not bulk:
		print "Time elapsed:", time.clock() - begintime, "seconds"
		print "@@@ found slit @@@"
    return slitstart,slitmiddle,slitend,centerfit

def safeslit(slitstart, slitmiddle, slitend):
    """
    This trims the found location of the slit to prevent things like accidentally
    counting part of the thick slit as thin, or counting areas outside the slit
    inside it
    """
    dif = slitend-slitstart
    newstart = int(slitstart + pct*dif)
    newend = int(slitend - pct*dif)
    
    newmiddle1 = int(slitmiddle - pct*dif)
    newmiddle2 = int(slitmiddle + pct*dif)
    
    arr = [newstart, newmiddle1, newmiddle2, newend]
    if arr != sorted(arr): #if the slit parameters are not in the proper order
        sys.tracebacklimit=0 #prevent traceback
        raise ValueError("Slit too small to create safe regions at pct = "+str(pct))
    else:
        return arr
    
def binselect(rowstart, rowmid, rowend, rgb, bincentres, wavelengths, testing=False, bulk=False):
    """
    This will decide which binning method is fastest, and then do that
    """
    if 3.0 * len(bincentres) > rgb.shape[2]: #after some testing, this turned out to be a reasonable
    #estimate for which method would be faster (it is the number of loop indices for binrgb2 vs binrgb, both
    #divided by the total number of rows to be binned)
        thinbins, thickbins = binrgb(rowstart, rowmid, rowend, rgb, bincentres, wavelengths, testing, bulk)
    else:
        thinbins, thickbins = binrgb2(rowstart, rowmid, rowend, rgb, bincentres, wavelengths, testing, bulk)
    return thinbins, thickbins
    
def binrgb(rowstart, rowmid, rowend, rgb, bincentres, wavelengths, testing=False, bulk=False):
    """
    This function will return resampled arrays
    """
    if not bulk:
		a = time.clock()
		print "\n@@@ resampling @@@"
    #thick and thin will contain [r, g, b]
    binwidth = bincentres[1]-bincentres[0]
    lb = len(bincentres)
    thick = np.tile(np.nan, (lb,3))
    thin = np.tile(np.nan, (lb,3))
    #we will ignore the region from(1-pct)*rowmid to (1+pct)*rowmid to prevent
    #accidentally adding thick to thin or vice versa
    rowmid1 = int(rowmid*(1.0-pct))
    rowmid2 = int(rowmid*(1.0+pct))
    thinall = np.tile(np.nan, (rowmid1-rowstart,lb,3))
    thickall = np.tile(np.nan, (rowend-rowmid2,lb,3))
    warnings.simplefilter("ignore") #suppress mean-of-empty-array warning
    
    #first we do the thin slit part
    if not testing and not bulk:
		print "Thin part - from", rowstart, "to", rowmid1
    for j in np.arange(rowstart, rowmid1)-rowstart: #row
        addcount = np.zeros((lb, 3)) #this counts how many data points we have added to one
        #bin, so that we can easily calculate the average at the end
        rgbadded = np.zeros((lb, 3))
        for i in range(rgb.shape[2]):
            #now we check every value in rgb and add it to the bin it should be in
            #l is the bin the value should be put in
            l = np.floor((wavelengths[j,i]+0.5*binwidth-bincentres[0])/binwidth)
            if l >= 0 and l < lb:
                addcount[l] += 1
                rgbadded[l,:] += rgb[:,j,i]
        rgbadded /= addcount
        thinall[j] = rgbadded
    thin = np.average(thinall,axis=0)

    #next we do the thick slit part
    if not testing and not bulk:
		print "Thick part - from", rowmid2, "to", rowend
    for j in np.arange(rowmid2, rowend)-rowstart: #row
        addcount = np.zeros((lb, 3)) #this counts how many data points we have added to one
        #bin, so that we can easily calculate the average at the end
        rgbadded = np.zeros((lb, 3))
        for i in range(rgb.shape[2]):
            #now we check every value in rgb and add it to the bin it should be in
            #l is the bin the value should be put in
            l = np.floor((wavelengths[j,i]+0.5*binwidth-bincentres[0])/binwidth)
            if l >= 0 and l < lb:
                addcount[l] += 1
                rgbadded[l,:] += rgb[:,j,i]
        rgbadded /= addcount
        thickall[j+rowstart-rowmid2] = rgbadded
	thick = np.average(thickall,axis=0)
    
	if not bulk:
		print "time elapsed:", time.clock()-a, "seconds"
		print "@@@ finished @@@"
    return thin, thick

def binrgb2(rowstart, rowmid, rowend, rgb, bincentres, wavelengths, testing=False, bulk=False):
	"""
	Re-write of binrgb(), which should be much faster
	"""
	if not bulk:
		a = time.clock()
		print "\n@@@ resampling @@@"
	binwidth = bincentres[1]-bincentres[0]
	rowmid1 = int(rowmid*(1.0-pct))
	rowmid2 = int(rowmid*(1.0+pct))
	ms = rowmid1-rowstart
	lb = len(bincentres)
	ll = range(0, lb)
    #we will ignore the region from(1-pct)*rowmid to (1+pct)*rowmid to prevent
    #accidentally adding thick to thin or vice versa
	thin = np.zeros((lb,3))
	thinall = np.zeros((rowmid1-rowstart, lb, 3))
	thick = np.zeros((lb,3))
	thickall = np.zeros((rowend-rowmid2, lb, 3))

	lall = np.floor((wavelengths + 0.5*binwidth-bincentres[0])/binwidth)
    #lall = what bin the data from each pixel should go in
    #first take the means per bin per row, then take the mean over all rows per bin
	warnings.simplefilter("ignore") #suppress mean-of-empty-array warning
	if not testing and not bulk:
		print "Thin part - from", rowstart, "to", rowmid1
	for j in range(ms):
		for l in ll:
			for c in [0,1,2]:
				thinall[j,l,c] = np.mean(rgb[c][j][np.where(lall[j] == l)[0]])
	thin = np.nanmean(thinall, axis=0)
	
	if not testing and not bulk:
		print "Thick part - from", rowmid2, "to", rowend
	for j in range(rowend-rowmid2):
		for l in ll:
			for c in [0,1,2]:
				thickall[j,l,c] = np.mean(rgb[c][j+ms][np.where(lall[j+ms] == l)[0]])
	thick = np.nanmean(thickall, axis=0)
	sys.tracebacklimit=1000
	if not bulk:
		print "time elapsed:", time.clock()-a, "seconds"
		print "@@@ finished @@@"
	return thin, thick

def thintothick(thinvalues, gamma, a):
	"""
	Fits thick-slit values to thin-slit values using an sRGB-like gamma curve
	Used for fitting the gamma function
	"""
	"""
	thickvalues = np.copy(thinvalues)
	smallerthanb = np.where(thinvalues < b)
	largerthanb = np.delete(np.arange(thinvalues.shape[0]), smallerthanb)
	thickvalues[smallerthanb] = thinvalues[smallerthanb]/c
	thickvalues[largerthanb] = (thinvalues[largerthanb] + a) * 2.0**(1.0/gamma) - a
	"""
	thickvalues = (thinvalues + a) * 2.0**(1.0/gamma) - a
	return thickvalues
	
def inversegamma(rgb, gamma, a):
	"""
	Inverse of the fitted gamma transformation
	"""
	rgb2 = rgb / 255.0
	anew = a
	rgblinear = ( (rgb2 + anew) / (1.0 + anew))**gamma
	return rgblinear

def normalisethickthin(thick, thin):
	"""
	Normalises a thick and thin slit spectrum by dividing them by the maximum
	of the thick part
	"""
	factor = np.nanmax(np.concatenate((thick[:])))
	return thick/factor, thin/factor

def rgbbins(bincentres):
	"""
	Gives the bins where we have determined (prior knowledge) the r,g,b pixels values
	to be relevant
	"""
	rbins = np.where(np.in1d(bincentres,bincentres[np.where(bincentres > 540)][np.where(bincentres[np.where(bincentres > 540)] < 680)]))
	gbins = np.where(np.in1d(bincentres,bincentres[np.where(bincentres > 450)][np.where(bincentres[np.where(bincentres > 450)] < 630)]))
	bbins = np.where(np.in1d(bincentres,bincentres[np.where(bincentres > 390)][np.where(bincentres[np.where(bincentres > 390)] < 550)]))
	return [rbins, gbins, bbins]	

def sRGBinverse(value):
    if value <= 10.31475: #0.04045 * 255.0
        return value/12.92
    else:
        return 255.0 * ( (value/255.0 + 0.055) / 1.055)**2.4

def sRGB(rgb):
    """
    Performs the simple sRGB gamma inverse transformation

    rgb is assumed to be a resampled array of the form [r, g, b]
    """
    #a = time.clock()
    #print "\n@@@ performing inverse gamma transformation @@@"
    rgb2 = np.copy(rgb) #so that we don't edit the original
    for i in range(rgb2.shape[0]):
        rgb2[i,0] = sRGBinverse(rgb2[i,0])
        rgb2[i,1] = sRGBinverse(rgb2[i,1])
        rgb2[i,2] = sRGBinverse(rgb2[i,2])
    #print "time elapsed:", time.clock() - a, "seconds\n@@@ finished @@@"
    return rgb2
