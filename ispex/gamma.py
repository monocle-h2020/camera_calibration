import numpy as np

polariser_angle = 74.14017
pixel_angle = 3120.
I_range = np.linspace(0, 1, 501)

def malus(angle):
    return (np.cos(np.radians(angle)))**2

def cos4f(d, f, a, o):
    return o + a*(np.cos(np.arctan(d/f)))**4

def find_I0(rgbg, distances, radius=100):
    return rgbg[distances < radius].mean()
