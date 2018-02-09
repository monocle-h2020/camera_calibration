"""
Olivier Burggraaff
Leiden University
Bacheloronderzoek: Smartphone Spectrometry

(Gamma), Filter, White balance calibration
"""

import time
begintime = time.clock()
print "py_sun py_sun py_sun py_sun py_sun py_sun\n"
import numpy as np
from matplotlib import pyplot as plt
import os
import py_ispex as ispex #Custom library with functions for these scripts
import sys
sys.tracebacklimit=1000 #traceback

binwidth = 8 #nm
bincentres = np.arange(330, 700, binwidth)

files = []
a = "sunn"
b = ".JPG"
for i in range(1,10):
    files.append(a+str(i)+b)

#imagename = "sunn6.JPG" #The image to be analysed
for imagename in files:
	imagenoext = os.path.splitext(imagename)[0] #without extension

	#Importing the image
	img = ispex.importimage(imagename)
	is0 = int(img.size[0]/2)
	is0h = img.size[0] - is0
	pixels = img.load()
	pct = 0.04
	rowstart1, rowmid, rowend1, slitpos = ispex.findslit2(pixels, img.size, imagenoext)
	rowstart = int(rowstart1 + pct*(rowend1-rowstart1))
	rowend = int(rowend1 - pct*(rowend1-rowstart1))
	ss = int(ispex.ssbp*img.size[1]) #we can assume the spectrum to be between rows
	se = int(ispex.ssep2*img.size[1]) #ss and se
	#we have to remove the entires from slitpos that we won't use (the ones between rowstart1 and rowstart, and between rowend and rowend1)
	slitpos = np.delete(slitpos, range(rowstart-rowstart1))
	slitpos = np.delete(slitpos, range(rowend-rowstart+1, slitpos.shape[0]+1))

	wavelengths = ispex.importwavelength(rowstart, rowend, img.size, slitpos, filename="fitparameters"+str(img.size[0])+"_"+str(img.size[1])+".txt")
	x2 = np.arange(rowstart, rowend+1)
	printperc = len(x2)/5
	rgb = np.tile(np.nan, (3, rowend+1-rowstart,is0h))
	print "\n@@@ importing r,g,b values for rows", rowstart, "-", rowend, "@@@"
	a = time.clock()
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
	print "time elapsed:", time.clock()-a, "seconds\n@@@ imported r,g,b @@@"

	thinbins, thickbins = ispex.binselect(rowstart, rowmid, rowend, rgb, bincentres, wavelengths)

	colours = ['red', 'green', 'blue']
	plt.figure(figsize=(20,10))
	for i in [0,1,2]:
		plt.plot(bincentres,thinbins[:,i], color=colours[i], lw=1, label='thin')
		plt.plot(bincentres,thickbins[:,i], color=colours[i], lw=3, label='thick')
	plt.legend(loc='upper right')
	plt.xlabel("wavelength (nm)")
	plt.ylabel("r,g,b value ; gamma corrected")
	plt.ylim(-1, 260)
	plt.xlim(bincentres[0], bincentres[-1])
	plt.axhline(256, color='black')
	plt.title("Resampling of file "+imagename+" ; binsize "+str(binwidth))
	#ispex.vertlines()
	plt.savefig(imagenoext+"_binned.png")
	plt.close()

	bbins = np.where(bincentres < 550)
	rbins = np.where(bincentres > 550)
	gbins = np.where(np.in1d(bincentres,bincentres[np.where(bincentres > 450)][np.where(bincentres[np.where(bincentres > 450)] < 630)]))
	rgbbins = [rbins, gbins, bbins]

	divided = thickbins/thinbins
	gamma = 1.0 / (np.log2(divided))

	thingamma = np.copy(thinbins)
	thickgamma = np.copy(thickbins)
	plt.figure(figsize=(20,10))
	plt.xlim(0,255)
	plt.ylim(0,255)
	plt.xlabel("Thin slit values")
	plt.ylabel("Corresponding thick slit values")
	#Least squares fit for gamma
	def thintothick(thinvalues, gamma, a):
		#thickvalues = np.tile(np.nan, len(thinvalues))
		#thickvalues = thinvalues * 2.0**(1.0/gamma)
		thickvalues = (thinvalues + a) * 2.0**(1.0/gamma) - a
		return thickvalues

	thintouse = []
	thicktouse = []
	for i in [0,1,2]:
		#we do it separately for every colour
		thinnan = np.where(np.isnan(thingamma[:,i]))[0]
		thicknan = np.where(np.isnan(thickgamma[:,i]))[0]
		c = np.where(thingamma/thickgamma > 1.0)[0]
		saturated = np.where(thickgamma > 245)[0]
		eithernan = np.unique(np.concatenate((thinnan, thicknan, c, saturated)))
		thingammaD = np.delete(thingamma[:,i], eithernan)
		thickgammaD = np.delete(thickgamma[:,i], eithernan)
		plt.scatter(thingammaD, thickgammaD, color=colours[i], label="Measured thick values")
		thicktouse.append(thickgammaD)
		thintouse.append(thingammaD)
	thicktouse = np.array(thicktouse).flatten()
	thintouse = np.array(thintouse).flatten()
	leastsquares = np.inf
	bestgamma = 2.4
	besta = 14.0
	for g in np.arange(0.0, 4.0, 0.01):
		for a in np.arange(0.0, 50.0, 0.1):
			residuals = thicktouse - thintothick(thintouse, g, a)
			S =  np.sum(residuals**2.0)
			if S < leastsquares:
				leastsquares = S
				bestgamma = g
				besta = a
	plt.plot(np.arange(0,256,1), thintothick(np.arange(0,256,1), bestgamma, besta), color='black', label='Fit')
	plt.title("Gamma fit: gamma = "+str(bestgamma)+" ; a = "+str(besta))
	plt.legend(loc='lower right')
	plt.savefig(imagenoext+"_gammafit.png")
	plt.close()
	print "Found fit: gamma =", bestgamma, "; a =", besta
	plt.figure(figsize=(20,10))
	plt.ylim(-1,260)
	plt.xlim(bincentres[0], bincentres[-1])
	plt.xlabel("wavelength (nm)")
	plt.ylabel("r/g/b value")
	plt.title("Thick values and fitted thick values (from thin ones)")
	for i in [0,1,2]:
		plt.plot(bincentres, thickbins[:,i], color=colours[i], lw=3)
		plt.plot(bincentres, thintothick(thinbins[:,i], bestgamma, besta), color=colours[i], lw=1)
	plt.savefig(imagenoext+"_gamma_correct.png")
	plt.close()

	def inversegamma(rgb, gamma, a):
		"""
		Inverse of the fitted gamma transformation
		"""
		"do something here please"
		rgb2 = rgb / 255.0
		anew = a / 255.0
		rgblinear = ( (rgb + anew) / (1 + anew))**gamma
		return rgblinear

	plt.figure(figsize=(20,10))
	intensityCK = np.tile(np.nan, (len(bincentres), 3))
	intensityN = np.copy(intensityCK)
	for i in [0,1,2]:
		intensityCK[:,i] = inversegamma(thickbins[:,i], bestgamma, besta)
		intensityN[:,i] = inversegamma(thinbins[:,i],bestgamma, besta)
	intensityN /= np.nanmax(intensityCK)
	intensityCK /= np.nanmax(intensityCK)
	for i in [0,1,2]:
		plt.plot(bincentres, intensityCK[:,i], color=colours[i], lw=3)
		plt.plot(bincentres, intensityN[:,i], color=colours[i], lw=1)
	plt.xlim(bincentres[0], bincentres[-1])
	plt.xlabel("wavelength (nm)")
	plt.ylabel("gamma-corrected rgb values")
	plt.title("RGB values, corrected using inverse of fitted gamma correction, normalised\nAverage factor is "+str(np.nanmean(intensityCK/intensityN)))
	plt.savefig(imagenoext+"_inverse_gamma.png")
	plt.close()
	"""
	filtercurves = np.tile(np.nan, (len(bincentres), 3))
	plt.figure(figsize=(20,10))
	for i in [0,1,2]:
		filtercurves[:,i] = np.nanmean([thintothick(thinbins[:,i], gammas[i], a_s[i]), thickbins[:,i]], axis=0)
		plt.plot(bincentres, filtercurves[:,i], color=colours[i], lw=3)
	plt.legend(loc='upper right')
	plt.xlabel("wavelength (nm)")
	plt.ylabel("filter response (arbitrary units)")
	plt.ylim(-1, 260)
	plt.xlim(bincentres[0], bincentres[-1])
	plt.axhline(256, color='black')
	plt.title("Filter curves as determined from image "+imagenoext+" ; binsize "+str(binwidth)+" nm\nAverage of thin and thick slit")
	#ispex.vertlines()
	plt.savefig(imagenoext+"_filtercurves.png")
	plt.close()
	"""
	"""
	fac = (np.nanmax(thickbins,axis=0))/(np.nanmax(thinbins,axis=0))
	plt.figure(figsize=(20,10))
	for i in [0,1,2]:
		plt.plot(bincentres,thinbins[:,i]*fac[i], color=colours[i], lw=1, label='thin')
		plt.plot(bincentres,thickbins[:,i], color=colours[i], lw=3, label='thick')
	plt.legend(loc='upper right')
	plt.xlabel("wavelength (nm)")
	plt.ylabel("r,g,b value ; gamma corrected")
	plt.ylim(-1, 260)
	plt.xlim(350, 700)
	plt.axhline(256, color='black')
	plt.title("Resampling of file "+imagename+" ; binsize "+str(binwidth)+"\nfactors: r "+str(fac[0])+" ; g "+str(fac[1])+" ; b "+str(fac[2]))
	#ispex.vertlines()
	plt.savefig(imagenoext+"_binned_fac.png")
	plt.close()

	shift = [0,0,0]
	for i in [0,1,2]:
		shift[i] = np.where(thickbins[:,i] == np.nanmax(thickbins[:,i]))[0][0] - np.where(thinbins[:,i] == np.nanmax(thinbins[:,i]))[0][0]
	shift = np.round(np.average(shift))

	sunbbspec = ispex.bb(ispex.T_sun, bincentres)
	plt.figure(figsize=(20,10))
	for i in [0,1,2]:
		plt.plot(bincentres,thinbins[:,i]*fac[i] / sunbbspec, color=colours[i], lw=1, label='thin')
		plt.plot(bincentres,thickbins[:,i]/ sunbbspec, color=colours[i], lw=3, label='thick')
	plt.legend(loc='upper right')
	plt.xlabel("wavelength (nm)")
	plt.ylabel("r,g,b value ; gamma corrected")
	plt.ylim(-1, 260)
	plt.xlim(350, 700)
	plt.axhline(256, color='black')
	plt.title("Resampling of file "+imagename+" divided by solar BB spectrum; binsize "+str(binwidth)+"\nfactors: r "+str(fac[0])+" ; g "+str(fac[1])+" ; b "+str(fac[2]))
	#ispex.vertlines()
	plt.savefig(imagenoext+"_binned_fac_bb.png")
	plt.close()

	filtercurves = np.tile(np.nan, (len(bincentres), 3))
	plt.figure(figsize=(20,10))
	for i in [0,1,2]:
		filtercurves[:,i] = np.nanmean([thinbins[:,i]*fac[i], thickbins[:,i]], axis=0)
		plt.plot(bincentres, filtercurves[:,i], color=colours[i], lw=3)
	plt.legend(loc='upper right')
	plt.xlabel("wavelength (nm)")
	plt.ylabel("filter response (arbitrary units)")
	plt.ylim(-1, 260)
	plt.xlim(350, 700)
	plt.axhline(256, color='black')
	plt.title("Filter curves as determined from image "+imagenoext+" ; binsize "+str(binwidth)+" nm\nAverage of thin and thick slit")
	#ispex.vertlines()
	plt.savefig(imagenoext+"_filtercurves.png")
	plt.close()

	np.savetxt("filtercurves.txt", np.array([bincentres, filtercurves[:,0], filtercurves[:,1], filtercurves[:,2]]).transpose())
	"""
	print "\nthis was file", imagename
print "Elapsed time:", -begintime + time.clock(), "seconds"
