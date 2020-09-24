import numpy as np


def _find_offset(color_pattern, colour):
    pos = np.array(np.where(color_pattern == colour)).T[0]
    return pos


def demosaick(bayer_map, *data, **kwargs):
    """
    Simplified demosaicking method for RGBG data.
    Uses a Bayer map `bayer_map` (RGBG channel for each pixel) and any number
    of input arrays `data`. Any additional **kwargs are passed to pull_apart.
    """
    # Demosaick the data
    data_RGBG = [pull_apart(data_array, bayer_map, **kwargs)[0] for data_array in data]

    # If only a single array was given, don't return a list
    if len(data_RGBG) == 1:
        data_RGBG = data_RGBG[0]

    return data_RGBG


def pull_apart(raw_img, color_pattern, color_desc="RGBG"):
    if color_desc not in ("RGBG", b"RGBG"):
        raise ValueError(f"Unknown colour description `{color_desc}")
    offsets = np.array([_find_offset(color_pattern, i) for i in range(4)])
    offX, offY = offsets.T
    R, G, B, G2 = [raw_img[x::2, y::2] for x, y in zip(offX, offY)]
    RGBG = np.stack((R, G, B, G2))
    return RGBG, offsets


def put_together_from_offsets(R, G, B, G2, offsets):
    result = np.zeros((R.shape[0]*2, R.shape[1]*2))
    for colour, offset in zip([R,G,B,G2], offsets):
        x, y = offset
        result[x::2, y::2] = colour
    result = result.astype(R.dtype)
    return result


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
