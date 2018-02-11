"""
Olivier Burggraaff
Leiden University
Bacheloronderzoek: Smartphone Spectrometry

Plotting of calibrated spectrum
"""

import time
begintime = time.clock()
print "~~~~~ STARTING SPECTRUM PLOTTING SCRIPT ~~~~~\n"
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import os
import ispex  # Custom library with functions for these scripts

fig = plt.figure(figsize=(20,30))
gs = gridspec.GridSpec(500, 300)
fig.suptitle("Spectra of halogen lamp of various temperatures, with black-body fits\nData in black, fit in red")
axs = [plt.subplot(gs[5:95, 5:95]), plt.subplot(gs[5:95, 105:195]), plt.subplot(gs[5:95, 205:295]), plt.subplot(gs[105:195, 5:95]), plt.subplot(gs[105:195, 105:195]), plt.subplot(gs[105:195, 205:295]), plt.subplot(gs[205:295, 5:95]), plt.subplot(gs[205:295, 105:195]), plt.subplot(gs[205:295, 205:295]), plt.subplot(gs[305:395, 5:95]), plt.subplot(gs[305:395, 105:195]), plt.subplot(gs[305:395, 205:295]), plt.subplot(gs[405:495, 5:95]), plt.subplot(gs[405:495, 105:195]), plt.subplot(gs[405:495, 205:295])]

F = -1
flist = np.sort(np.concatenate((ispex.get_filelist("E:\\BO\\Temperature\\", "2"), ispex.get_filelist("E:\\BO\\Temperature\\", "1"))))
for imagename in flist:
    imagenoext = os.path.splitext(imagename)[0]
    print os.path.split(imagenoext)[1],
    img = ispex.importimage(imagename, bulk=True)
    gamma = ispex.importgamma("gamma.txt")
    bincentres, filtercurves = ispex.importfilters("filtercurves.txt")

    F += 1

    axs[F].set_xlim(bincentres[0], bincentres[-1])
    axs[F].set_ylim(-0.02, 1.02)
    axs[F].grid(True)

    if F%3 == 0:
        axs[F].set_ylabel('Intensity (norm.)')

    else:
        axs[F].tick_params(axis='y', labelleft='off')

    if F/3 == 4:
        axs[F].set_xlabel('Wavelength (nm)')

    else:
        axs[F].tick_params(axis='x', labelbottom='off')

    is0 = int(img.size[0]/2)
    is0h = img.size[0] - is0
    pixels = img.load()
    ss = int(ispex.slitstartpct[0]*img.size[1]) #we can assume the spectrum to be between rows ss and se
    se = int(ispex.slitendpct[1]*img.size[1])
    rowstart, rowmid, rowend, slitpos = ispex.findslit(img)
    wvlfile = os.path.dirname(imagename)+"\\fitparameters"+str(img.size[0])+"_"+str(img.size[1])+".txt"
    wavelengths = ispex.importwavelength(rowstart, rowend, img.size, slitpos, wvlfile, bulk=True)

    rgb = np.tile(np.nan, (3,rowend+1-rowstart,is0h))
    for j in range(rowstart, rowend+1): #row
    	r = np.tile(np.nan, is0h)
    	g = np.tile(np.nan, is0h)
    	b = np.tile(np.nan, is0h) #we will store the r,g,b values per row in these arrays
    	for i in range(is0h): #column
    		r[i], g[i], b[i] = pixels[i+is0,j]

    	rgb[:,j-rowstart] = r,g,b

    rgbg = ispex.invgamma(rgb, gamma)

    thinbins, thickbins = ispex.resample(rowstart, rowmid, rowend, rgbg, bincentres, wavelengths)

    filtered = np.tile(np.nan, (len(bincentres), 3))
    rgbbins = ispex.rgbbins(bincentres)

    for c in [0,1,2]:
        filtered[rgbbins[c],c] = np.nanmean([ispex.ratio*thinbins[rgbbins[c],c], thickbins[rgbbins[c], c]], axis=0) / filtercurves[rgbbins[c],c]
        filtered[:,c] /= np.nanmax(filtered[:,c])

    wb = np.array([np.nan, 1.0, np.nan])
    wb[0] = np.nanmean(filtered[:,1]/filtered[:,0])
    wb[2] = np.nanmean(filtered[:,1]/filtered[:,2])

    #we will take the weighted averages and put them in the array toplot
    bgs = rgbbins[1][0]
    bge = rgbbins[2][-1]
    bga = np.arange(0.0, bge-bgs)
    grs = rgbbins[0][0]
    gre = rgbbins[1][-1]
    gra = np.arange(0.0, gre-grs)
    toplot = np.tile(np.nan, bincentres.shape)
    toplot[rgbbins[2][0] : bgs] = filtered[rgbbins[2][0] : bgs , 2] * wb[2] #before green starts
    toplot[bge : grs] = filtered[bge : grs, 1] * wb[1] #after blue ends, before red starts
    toplot[gre : rgbbins[0][-1]] = filtered[gre : rgbbins[0][-1], 0] * wb[0] #after green ends

    #blue and green
    toplot[bgs:bge] = (wb[1] * filtered[bgs:bge, 1] * bga + wb[2] * filtered[bgs:bge, 2] * bga[::-1]) / (bga + bga[::-1])

    #green and red
    toplot[grs:gre] = (wb[0] * filtered[grs:gre, 0] * gra + wb[1] * filtered[grs:gre, 1] * gra[::-1]) / (gra + gra[::-1])

    bestT = 5000
    bestS = np.inf

    nonnan = np.where(-np.isnan(toplot))
    t = toplot[nonnan]
    b = bincentres[nonnan]
    #t = t[b < 600]
    #b = b[b < 600]
    t /= np.nanmax(t)


    for T in range(500, 6000, 10):
        S = np.sum((ispex.bb(b, T) - t)**2.0)
        if S < bestS:
            bestS = S
            bestT = T

    plt.figure(figsize=(20,10))
    plt.plot(bincentres, toplot/np.nanmax(toplot), c='k', lw=2, label='data')
    plt.plot(b, ispex.bb(b, bestT), c='r', lw=2, label='BB fit')
    plt.legend(loc='upper left')
    plt.xlim(380, 700)
    plt.ylim(-0.02, 1.05)
    plt.xlabel("wavelength (nm)")
    plt.ylabel("intensity (a.u.)")
    plt.title("Final spectrum of file "+imagename+" ; black-body temperature "+str(bestT)+" K")
    plt.savefig(imagenoext+"_BB_fit.png")
    plt.close()

    axs[F].plot(bincentres, toplot/np.nanmax(toplot), c='k', lw=1, label="T = "+os.path.split(imagenoext)[1][:-2]+" K")
    axs[F].plot(b, ispex.bb(b, bestT), c='r', lw=1, label="T = "+str(bestT)+" K")
    axs[F].legend(loc='upper left')

fig.savefig(os.path.split(flist[0])[0]+"\\BB.png")
plt.close()