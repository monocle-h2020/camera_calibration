"""
Olivier Burggraaff
Leiden University
Bacheloronderzoek: Smartphone Spectrometry

Gamma calibration
"""

import time
begintime = time.clock()
print "py_gamma py_gamma py_gamma py_gamma py_gamma\n"
import numpy as np
from matplotlib import pyplot as plt
import os
import sys
import py_ispex as ispex #Custom library with functions for these scripts
from scipy import optimize as opt

binwidth = 8 #nm
bincentres = np.arange(380, 700, binwidth)
rgbbins = ispex.rgbbins(bincentres)
colours = ['red', 'green', 'blue']

allthin = []
allthick = []
everythin = []
everythick = []
#firstthin = []
#firstthick = []

plt.figure(figsize=(20,10))
plt.xlim(0,255)
plt.ylim(0,255)
plt.xlabel("Thin slit values")
plt.ylabel("Corresponding thick slit values")

print "\n@@@ importing data @@@"
#ispex.get_filelist("/home/burggraaff/BO/", "sunn")
#filelist = []
#for i in range(10,51):
#	filelist.append("sunn"+str(i)+".JPG")
for imagename in ispex.get_filelist("E:\BO\\", ""):
	print imagename,
	sys.stdout.flush()
	imagenoext = os.path.splitext(imagename)[0] #without extension

	#Importing the image
	img = ispex.importimage(imagename, bulk=True)
	is0 = int(img.size[0]/2)
	is0h = img.size[0] - is0
	pixels = img.load()
	pct = 0.04
	rowstart1, rowmid, rowend1, slitpos = ispex.findslit2(pixels, img.size, imagenoext, bulk=True)
	rowstart = int(rowstart1 + pct*(rowend1-rowstart1))
	rowend = int(rowend1 - pct*(rowend1-rowstart1))
	ss = int(ispex.ssbp*img.size[1]) #we can assume the spectrum to be between rows
	se = int(ispex.ssep2*img.size[1]) #ss and se
	#we have to remove the entires from slitpos that we won't use (the ones between rowstart1 and rowstart, and between rowend and rowend1)
	slitpos = np.delete(slitpos, range(rowstart-rowstart1))
	slitpos = np.delete(slitpos, range(rowend-rowstart+1, slitpos.shape[0]+1))

	wavelengths = ispex.importwavelength(rowstart, rowend, img.size, slitpos, filename="fitparameters"+str(img.size[0])+"_"+str(img.size[1])+".txt", bulk=True)
	x2 = np.arange(rowstart, rowend+1)
	printperc = len(x2)/5
	rgb = np.tile(np.nan, (3, rowend+1-rowstart,is0h))
	for j in x2: #row
		r = np.tile(np.nan, is0h)
		g = np.tile(np.nan, is0h)
		b = np.tile(np.nan, is0h) #we will store the r,g,b values per row in these arrays
		for i in range(is0h): #column
			r[i], g[i], b[i] = pixels[i+is0,j]
			#inverse gamma transformation:
			#r[i] = ispex.sRGBinverse(r[i])
			#g[i] = ispex.sRGBinverse(g[i])
			#b[i] = ispex.sRGBinverse(b[i])

		rgb[:,j-rowstart] = r, g, b

	thinbins, thickbins = ispex.binselect(rowstart, rowmid, rowend, rgb, bincentres, wavelengths, bulk=True)

	for i in [0,1,2]:
		#we do it separately for every colour
		thingamma = thinbins[:,i][rgbbins[i]]
		thickgamma = thickbins[:,i][rgbbins[i]]
		thinnan = np.where(np.isnan(thingamma))[0]
		thicknan = np.where(np.isnan(thickgamma))[0]
		#ratiotoolow = np.where(thingamma/thickgamma > 1.0)[0]
		#ratiotoohigh = np.where(thickgamma/thingamma > 3.5)[0]
		#saturated = np.where(thickgamma > 220)[0]
		#toolow = np.where(thingamma < 20)[0]
		#nottoolow = np.where(thingamma >= 20)[0]
		#removethese = np.unique(np.concatenate((thinnan, thicknan, saturated)))
		#removeforfirstsection = np.unique(np.concatenate((thinnan, thicknan, saturated, nottoolow)))
		#allremoved = np.unique(np.concatenate((thinnan, thicknan, saturated)))
		#thinfirst = np.delete(thingamma, removeforfirstsection)
		#thickfirst = np.delete(thickgamma, removeforfirstsection)
		#thingammaD = np.delete(thingamma, removethese)
		#thickgammaD = np.delete(thickgamma, removethese)
		#plt.scatter(thingammaD, thickgammaD, marker='o', color=colours[i])
		#plt.scatter(thinfirst, thickfirst, marker='s', color=colours[i])
		#plt.scatter(thingamma[allremoved], thickgamma[allremoved], color='black', marker='+')
		#firstthin.append(thinfirst)
		#firstthick.append(thickfirst)
		#allthin.append(thingammaD)
		#allthick.append(thickgammaD)
		thingamma = np.delete(thingamma, np.unique(np.concatenate((thinnan, thicknan))))
		thickgamma = np.delete(thickgamma, np.unique(np.concatenate((thinnan, thicknan))))
		everythin.append(thingamma)
		everythick.append(thickgamma)
		plt.scatter(thingamma, thickgamma, color=colours[i], marker='+')

#firstthin = np.concatenate((np.array(firstthin)[:]))
#firstthick = np.concatenate((np.array(firstthick)[:]))
#allthin = np.concatenate((np.array(allthin)[:]))
#allthick = np.concatenate((np.array(allthick)[:]))
everythin = np.concatenate((np.array(everythin)[:]))
everythick = np.concatenate((np.array(everythick)[:]))
"""
print "\n@@@ fitting gamma @@@"
leastsquares = np.inf
bestgammafirst = 1.5
bestgammamain = 2.4
bestamain = 0.0
bestsplit = 20

for split in np.arange(0,254,1):
	firstthin = allthin[np.where(allthin < split)]
	firstthick = allthick[np.where(allthin < split)]
	mainthin = allthin[np.where(allthin >= split)]
	mainthick = allthick[np.where(allthin >= split)]
	for g1 in np.arange(1.0, 2.0, 0.1): #bestgammafirst
		residualsfirst = firstthick/255.0 - ispex.thintothick(firstthin/255.0, g1, 0.0)
		Sfirst = np.sum(residualsfirst**2.0)
		for g2 in np.arange(1.0, 2.5, 0.1): #bestgammamain
			for a in np.arange(0.0, 0.5, 0.01): #bestamain
				residualsmain = mainthick/255.0 - ispex.thintothick(mainthin/255.0, g2, a)
				S = Sfirst + np.sum(residualsmain**2.0)
				if S < leastsquares:
					leastsquares = S
					bestgammafirst = g1
					bestgammamain = g2
					bestamain = a
					bestsplit = split

firstthin = allthin[np.where(allthin < bestsplit)]
firstthick = allthick[np.where(allthin < bestsplit)]
mainthin = allthin[np.where(allthin >= bestsplit)]
mainthick = allthick[np.where(allthin >= bestsplit)]
#for g in np.arange(0.0, 4.0, 0.01):
#	residuals = allthick/255.0 - ispex.thintothick(allthin/255.0, g, 0.0)
#	S = np.sum(residuals**2.0)
#	if S < leastsquares:
#		leastsquares = S
#		bestgamma = g
#print "Best fit: gamma =", bestgamma, "; S =", leastsquares
#
#print "\n@@@ fitting gamma and a @@@"
#for g in np.arange(1.1, 2.5, 0.01):
#	for a in np.arange(0.0, 0.5, 0.001):
#		residuals = allthick/255.0 - ispex.thintothick(allthin/255.0, g, a)
#		S = np.sum(residuals**2.0)
#		if S < leastsquares:
#			leastsquares = S
#			bestgamma = g
#
#print "Best fit: gamma =", bestgamma, "a =", besta, "; S =", leastsquares
#
#print "\n@@@ fitting gamma for first section @@@"
#leastsquares = np.inf
#firstgamma = 2.4
#firsta = 0.0
#for g in np.arange(0.0, 4.0, 0.01):
#	residuals = firstthick/255.0 - ispex.thintothick(firstthin/255.0, g, 0.0)
#	S = np.sum(residuals**2.0)
#	if S < leastsquares:
#		leastsquares = S
#		firstgamma = g
#print "Best fit: gamma =", firstgamma, "; S =", leastsquares
#
#print "\n@@@ fitting gamma and a @@@"
#for g in np.arange(1.1, 2.5, 0.01):
#	for a in np.arange(0.0, 0.5, 0.001):
#		residuals = firstthick/255.0 - ispex.thintothick(firstthin/255.0, g, a)
#		S = np.sum(residuals**2.0)
#		if S < leastsquares:
#			leastsquares = S
#			firstgamma = g
#			firsta = a
#print "Best fit: gamma =", firstgamma, "a =", firsta, "; S =", leastsquares
#
"""
plt.scatter([500], [500], c='k', marker='+', label='r/g/b data')
#plt.scatter(mainthin, mainthick, c='k', marker='o', label='r/g/b data used for main fit')
#plt.scatter(firstthin, firstthick, c='k', marker='s', label='r/g/b data used for first section fit')
#plt.plot(np.linspace(0.0, 255.0, 3), 1.0*np.linspace(0.0, 255.0, 3), c='y', label='constraints on data')
#plt.plot(np.linspace(0.0, 255.0, 3), 3.5*np.linspace(0.0, 255.0, 3), c='y')
#plt.axhline(220, c='y')
#plt.axvline(bestsplit, c='y')
#plt.plot(np.arange(0, bestsplit,1), ispex.thintothick(np.arange(0,bestsplit,1), bestgammafirst, 0.0), color='blue', label='First fit')
#plt.plot(np.arange(bestsplit,256,1), ispex.thintothick(np.arange(bestsplit,256,1), bestgammamain, bestamain*255.0), color='red', label='Main fit')
plt.grid(True)
plt.title("Thick slit rgb values for a given thin slit rgb value")
plt.legend(loc='lower right')

medians = np.tile(np.nan, 256)
means = np.tile(np.nan, 256)
mins = np.tile(np.nan, 256)
maxes = np.tile(np.nan, 256)
thins = np.arange(0, 256, 1)
for i in thins:
	elements = np.where(np.in1d(everythin,everythin[np.where(everythin > i-0.5)][np.where(everythin[np.where(everythin > i-0.5)] < i+0.5)]))[0]
	medians[i] = np.median(everythick[elements])
	means[i] = np.nanmean(everythick[elements])
	if elements.shape[0] != 0:
		mins[i] = np.nanmin(everythick[elements])
		maxes[i] = np.nanmax(everythick[elements])

"""
lowest = np.inf
besta = 8.0
for a in np.arange(0.1, 20.0, 0.1):
	residuals = means[1:] - np.log(np.arange(1.0, 256.0))/np.log(a)
	S = np.nansum(residuals**2.0)
	if S < lowest:
		besta = a
		lowest = S
a = besta
"""

def tofit(thin, a, b, c, d, e):
	return np.maximum(np.minimum(np.polyval([a,b,c,d,e,0.0],thin), np.tile(255.0, len(thin))), np.zeros(len(thin)))
"""
plt.close()

def funk(values, a,b,c,d,e,f):
	return a/values**4.0 + b/values**3.0 + c/values**2.0 + d/values + e + f*values
	# + np.polyval([b,c,d,e,0.0], values)

lowest = np.inf
besta = 1.0
bestb = 1.0
bestc = 1.0
bestd = 1.0
beste = 1.0
bestf = 1.0
for a in np.append(np.linspace(-0.7, -0.1, num=13),0):
	print a,
	for b in np.append(np.linspace(-100.0, 100.0, num=13),0):
		for c in np.append(np.linspace(-50.0, 100.0, num=13),0):
			for d in np.append(np.linspace(-250.0, 250.0, num=13),0):
				for e in np.append(np.linspace(0.0, 250.0, num=8),0):
					for f in np.append(np.linspace(0.0, 7.0, num=9),0):
						residuals = 2.0 - funk(everythick, a,b,c,d,e,f)/funk(everythin, a,b,c,d,e,f)
						S = np.nansum(residuals**2.0)
						if S < lowest:
							print S
							besta = a
							bestb = b
							bestc = c
							bestd = d
							beste = e
							bestf = f
							lowest = S
a = besta
b = bestb
c = bestc
d = bestd
e = beste
f = bestf
S = lowest
print [a,b,c,d,e,f,S]

plt.scatter(funk(everythick, a,b,c,d,e,f), funk(everythick, a,b,c,d,e,f)/funk(everythin, a,b,c,d,e,f), marker='+')
plt.ylim(0,3)
#means[0] = 0.0
#means[-1] = 255
#dele = np.where(np.isnan(medians))[0]
#weights = np.ones(means.shape[0] - dele.shape[0])
#weights[0] = 1000000
#weights[-1] = 1000000
#i = 5
#p = np.polyfit(np.delete(thins, dele), np.delete(medians, dele), i, w=weights)
#plt.plot(thins, np.polyval(p, thins), label=str(i)+"th degree polynomial fit to medians")
#plt.close()

thinthick = np.array([everythin, everythick])

def tofit2(thinandthick, a, b, c, d):
	return np.polyval([a,b,c, d,0.0], thinandthick[1])/np.polyval([a,b,c, d,0.0], thinandthick[0])

popt, pcov = opt.curve_fit(tofit2, thinthick, np.tile(2.0, len(everythin)), maxfev=1000000)
"""
popt, pcov = opt.curve_fit(tofit, everythin, everythick)
#plt.scatter(everythin, everythick/everythin, marker='+')


def fitted(thin):
	return tofit(thin, popt[0], popt[1], popt[2], popt[3], popt[4])
plt.plot(np.arange(256), fitted(np.arange(256)), color='black', lw=3)
plt.plot(thins, fitted(thins), color='black')
plt.savefig("Gammavalues.png")
plt.close()

plt.figure(figsize=(20,10))
#plt.plot(thins, np.polyval(p, thins), label=str(i)+"th degree polynomial fit to medians")
plt.scatter(thins, means)
plt.plot(thins, fitted(thins), color='black')
plt.xlim(0, 255)
plt.ylim(0, 255)
plt.xlabel("Thin slit value")
plt.ylabel("Means corresponding thick slit value")
plt.title("Gamma mapping - mean thick slit value for given thin slit value")
plt.legend(loc='lower right')
plt.savefig("Gammameans.png")
plt.close()

plt.figure(figsize=(20,10))
plt.scatter(thins, medians)
plt.plot(thins, fitted(thins), color='black')
#plt.plot(thins, np.polyval(p, thins), label=str(i)+"th degree polynomial fit to medians")
plt.xlim(0, 255)
plt.ylim(0, 255)
plt.xlabel("Thin slit value")
plt.ylabel("Medians corresponding thick slit value")
plt.title("Gamma mapping - median thick slit value for given thin slit value")
plt.savefig("Gammamedians.png")
plt.close()

print "\nElapsed time:", -begintime + time.clock(), "seconds"
