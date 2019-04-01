import numpy as np
from matplotlib import pyplot as plt
from phonecal.general import gauss1d

wvl, with_grating    = np.genfromtxt("reference_spectra/with_grating.txt"   , skip_header=14, unpack=True)
wvl, without_grating = np.genfromtxt("reference_spectra/without_grating.txt", skip_header=14, unpack=True)

with_grating    = gauss1d(with_grating   , 5)
without_grating = gauss1d(without_grating, 5)

with_grating    -= with_grating   [wvl < 350].mean()  # background subtraction
without_grating -= without_grating[wvl < 350].mean()

transmission_raw = with_grating / without_grating

wavelengths = np.arange(390, 702, 2)
transmission = np.interp(wavelengths, wvl, transmission_raw)

plt.plot(wvl, with_grating, c='r')
plt.plot(wvl, without_grating, c='k')
plt.show()

plt.plot(wavelengths, transmission)
plt.xlim(390, 700)
plt.ylim(0., 1.)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Transmission")
plt.savefig("results/grating_transmission.pdf")
plt.show()

trans = np.stack([wavelengths, transmission])

np.save("reference_spectra/transmission.npy", trans)
