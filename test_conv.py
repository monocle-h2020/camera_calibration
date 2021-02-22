from spectacle import load_camera
from spectacle.spectral import convolve, convolve_multi

from astropy import table
import numpy as np
from matplotlib import pyplot as plt

camera = load_camera(r"C:/Users/Burggraaff/SPECTACLE_data/iPhone_SE/")
camera._load_spectral_response()
print(camera)

path_sorad = r"C:\Users\Burggraaff\GitHub\water-colour\water-colour-data\Balaton_20190703\SoRad\So-Rad_Balaton2019.csv"
table_sorad = table.Table.read(path_sorad)

Rrs_cols = [col for col in table_sorad.keys() if "Rrs" in col]
wavelengths = np.array([float(col.split("_")[1]) for col in Rrs_cols])

Rrs_data = np.array(table_sorad[Rrs_cols]).view(np.float64).reshape((-1, len(wavelengths)))

Rrs0_convolvedR = convolve(camera.spectral_response[0], camera.spectral_response[1], wavelengths, Rrs_data[0])

RrsAll_convolvedR = convolve_multi(camera.spectral_response[0], camera.spectral_response[1], wavelengths, Rrs_data)

RrsAll_convolvedAll = np.array([convolve_multi(camera.spectral_response[0], srf, wavelengths, Rrs_data) for srf in camera.spectral_response[1:5]])

Rrs0_convolvedAll = camera.convolve(wavelengths, Rrs_data[0])

RrsAll_convolvedAll_camera = camera.convolve_multi(wavelengths, Rrs_data)
print("Difference between manual and camera methods:", RrsAll_convolvedAll_camera - RrsAll_convolvedAll)
