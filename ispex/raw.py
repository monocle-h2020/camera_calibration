import numpy as np

range_x = (1150, 1750)
range_y = (380, 735)  # change 380 to 550 for thick-only
x = np.arange(*range_x)

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


def to_RGB_array(raw_image, color_pattern):
    RGB = np.zeros((*raw_image.shape, 3))
    R_ind = np.where(color_pattern == 0)
    G_ind = np.where((color_pattern == 1) | (color_pattern == 3))
    B_ind = np.where(color_pattern == 2)
    RGB[R_ind[0], R_ind[1], 0] = raw_image[R_ind]
    RGB[G_ind[0], G_ind[1], 1] = raw_image[G_ind]
    RGB[B_ind[0], B_ind[1], 2] = raw_image[B_ind]
    return RGB


def cut_out_spectrum(RGBG):
    cut = RGBG[range_y[0]:range_y[1], range_x[0]:range_x[1], :]
    return cut
