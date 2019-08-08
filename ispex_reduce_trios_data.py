import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime
from spectacle import io
from sys import argv

def get_time(c):
    try:
        t = datetime.strptime(c, "%Y-%m-%dT%H:%M:%S.%f")
    except ValueError:
        try:
            t = datetime.strptime(c, "%Y-%m-%dT%H:%M:%S")
        except ValueError:
            t = np.nan
    try:
        return t.timestamp()
    except:
        return t

def to_float(x):
    try:
        return float(x)
    except ValueError:
        return np.nan

file = io.path_from_input(argv)
cal = np.genfromtxt(file, delimiter=",", dtype=str)

wavelengths = np.arange(320, 956.9, 3.3)

# 0: Trigger_time_PC
# 1: Trigger_time_GPS
# 2: Latitude
# 3: Longitude
# 4: Speed
# 5: Heading
# 6: Sensor_ID
# 7: Integration_time

table = np.tile(np.nan, cal.shape)
table[:, 0] = [get_time(c) for c in cal[:, 0]]
table[:, 1] = [get_time(c) for c in cal[:, 1]]
for j in range(2,6):
    table[:, j] = [to_float(x) for x in cal[:, j]]
# IDs: 20728 = sky-radiance Ls ; 34251 = downwelling Ed ; 34252 = upwelling Lu
table[:, 6] = [int(ID, 16) for ID in cal[:, 6]]
table[:, 7:] = cal[:, 7:]

upwelling = table[table[:, 6] == 34252.]

np.save(file.parent/(file.stem+".npy"), upwelling)

interpolated_wavelengths = np.arange(390, 701, 1)
new_upwelling = np.tile(np.nan, (len(upwelling), 8+len(interpolated_wavelengths)))
new_upwelling[:, :8] = upwelling[:, :8]
for j, row_old in enumerate(upwelling):
    new_upwelling[j, 8:] = np.interp(interpolated_wavelengths, wavelengths, row_old[8:])

np.save(file.parent/(file.stem+"_interpolated.npy"), new_upwelling)
