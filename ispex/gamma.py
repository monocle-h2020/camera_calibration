import numpy as np

degrad = np.pi/180
polariser_angle = 74.14017
pixel_angle = np.array([2825.46078149, 3205.27866603, 3120.66102908])
I_range = np.linspace(0, 1, 501)

def malus(angle, offset=polariser_angle):
    return (np.cos(np.radians(angle-offset)))**2

def malus_error(angle0, angle1=polariser_angle, I0=1., sigma_angle0=2., sigma_angle1=0.5, sigma_I0=0.01):
    alpha = angle0 - angle1
    A = I0 * degrad * np.sin(np.pi/90 * (alpha))
    s_a2 = A**2 * (sigma_angle0**2 + sigma_angle1**2)
    s_I2 = (malus(angle0, offset=angle1) * sigma_I0)**2
    total = np.sqrt(s_I2 + s_a2)

    return total

def cos4f(d, f, a, o):
    return o + a*(np.cos(np.arctan(d/f)))**4

def find_I0(rgbg, distances, radius=100):
    return rgbg[distances < radius].mean()
