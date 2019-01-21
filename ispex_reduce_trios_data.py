import numpy as np
from matplotlib import pyplot as plt
from datetime import datetime

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

cal = np.genfromtxt("test_files/Loch/20180821_MONOCLE_iSPEX_cal_01.txt", delimiter=",", dtype=str)
cal[cal == ""] = "nan"
cal[..., 6] = [int(a, 16) for a in cal[..., 6]]
cal[..., 0] = [get_time(c) for c in cal[...,0]]
cal[..., 1] = [get_time(c) for c in cal[...,1]]
cal2 = cal.astype(np.float64)

reflected = cal[::3]
