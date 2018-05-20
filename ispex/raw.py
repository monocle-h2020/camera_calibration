import numpy as np


def _find_offset(color_pattern, colour):
    pos = np.array(np.where(color_pattern == colour)).T[0]
    return pos


def pull_apart(raw_img, color_pattern, color_desc=b"RGBG"):
    if color_desc != b"RGBG":
        raise ValueError(f"Image is of type {raw_img.color_desc} instead of RGBG")
    offsets = np.array([_find_offset(color_pattern, i) for i in range(4)])
    offX, offY = offsets.T
    R, G, B, G2 = [raw_img[x::2, y::2] for x, y in zip(offX, offY)]
    RGBG = np.dstack((R, G, B, G2))
    return RGBG, offsets


def put_together(R, G, B, G2, offsets):
    result = np.zeros((R.shape[0]*2, R.shape[1]*2))
    for colour, offset in zip([R,G,B,G2], offsets):
        x, y = offset
        result[x::2, y::2] = colour
    result = result.astype(R.dtype)
    return result


def split_RGBG(RGBG):
    R, G, B, G2 = RGBG.T
    R, G, B, G2 = R.T, G.T, B.T, G2.T
    return R, G, B, G2
