import numpy as np


def _generate_bayer_slices(color_pattern, colours=range(4)):
    """
    Generate the slices used to demosaick data.
    """
    # Find the positions of the first element corresponding to each colour
    positions = [np.array(np.where(color_pattern == colour)).T[0] for colour in colours]

    # Make a slice for each colour
    slices = [np.s_[..., x::2, y::2] for x, y in positions]

    return slices


def demosaick(bayer_map, data, color_desc="RGBG"):
    """
    Uses a Bayer map `bayer_map` (RGBG channel for each pixel) and any number
    of input arrays `data`.
    """
    # Cast the data to a numpy array for the following indexing tricks to work
    data = np.array(data)

    # Check that we are dealing with RGBG2 data, as only these are supported right now.
    assert color_desc in ("RGBG", b"RGBG"), f"Unknown colour description `{color_desc}"

    # Check that the data and Bayer pattern have similar shapes
    assert data.shape[-2:] == bayer_map.shape, f"The data ({data.shape}) and Bayer map ({bayer_map.shape}) have incompatible shapes"

    # Demosaick the data along their last two axes
    bayer_pattern = bayer_map[:2, :2]
    slices = _generate_bayer_slices(bayer_pattern)

    # Combine the data back into one array of shape [..., 4, x/2, y/2]
    newshape = list(data.shape[:-2]) + [4, data.shape[-2]//2, data.shape[-1]//2]
    RGBG = np.empty(newshape)
    for i, s in enumerate(slices):
        RGBG[..., i, :, :] = data[s]

    return RGBG


def put_together_from_colours(RGBG, colours):
    original = np.zeros((2*RGBG.shape[1], 2*RGBG.shape[2]))
    for j in range(4):
        original[np.where(colours == j)] = RGBG[j].ravel()
    return original


def split_RGBG(RGBG):
    R, G, B, G2 = RGBG
    return R, G, B, G2


def to_RGB_array(raw_image, color_pattern):
    RGB = np.zeros((*raw_image.shape, 3))
    R_ind = np.where(color_pattern == 0)
    G_ind = np.where((color_pattern == 1) | (color_pattern == 3))
    B_ind = np.where(color_pattern == 2)
    RGB[R_ind[0], R_ind[1], 0] = raw_image[R_ind]
    RGB[G_ind[0], G_ind[1], 1] = raw_image[G_ind]
    RGB[B_ind[0], B_ind[1], 2] = raw_image[B_ind]
    return RGB


def multiply_RGBG(data, colours, factors):
    data_new = data.copy()
    for j in range(4):
        data_new[colours == j] *= factors[j]
    return data_new
