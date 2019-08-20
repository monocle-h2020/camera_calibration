import numpy as np

def load_gain_map(root):
    """
    Load the gain map located at `root`/results/gain_map.npy
    """
    gain_map = np.load(root/"results/gain_map.npy")
    return gain_map
