import numpy as np

def malus(angle, offset=polariser_angle):
    return (np.cos(np.radians(angle-offset)))**2

def malus_error(angle0, angle1=polariser_angle, I0=1., sigma_angle0=2., sigma_angle1=0.1, sigma_I0=0.01):
    alpha = angle0 - angle1
    A = I0 * degrad * np.sin(np.pi/90 * (alpha))
    s_a2 = A**2 * (sigma_angle0**2 + sigma_angle1**2)
    s_I2 = (malus(angle0, offset=angle1) * sigma_I0)**2
    total = np.sqrt(s_I2 + s_a2)

    return total
