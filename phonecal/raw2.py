import numpy as np

xmin = 2150
xmax = 3900
ymin = 760
ymax = 1470
x = np.arange(xmin, xmax)
y = np.arange(ymin, ymax)
x_small = np.arange(xmin/2, xmax/2)
y_small = np.arange(ymin/2, ymax/2)

def _find_offset(color_pattern, colour):
    pos = np.array(np.where(color_pattern == colour)).T[0]
    return pos


def pull_apart(raw_img, color_pattern, color_desc=b"RGBG"):
    if color_desc != b"RGBG":
        raise ValueError(f"Image is of type {raw_img.color_desc} instead of RGBG")
    offsets = np.array([_find_offset(color_pattern, i) for i in range(4)])
    offX, offY = offsets.T
    R, G, B, G2 = [raw_img[x::2, y::2] for x, y in zip(offX, offY)]
    RGBG = np.stack((R, G, B, G2))
    return RGBG, offsets


def pull_apart2(raw_img, color_pattern, color_desc="RGBG", remove=True):
    unique_colours = np.unique(color_pattern)

    stack = np.tile(np.nan, (*unique_colours.shape, *raw_img.shape))
    for j in range(len(unique_colours)):
        indices = np.where(color_pattern == j)
        stack[j][indices] = raw_img[indices]

    if remove and len(set(color_desc)) != len(color_desc):
        to_remove = []
        for j in range(1, len(unique_colours)):
            colour = color_desc[j]
            previous = color_desc[:j]
            try:
                ind_previous = previous.index(colour)
            except ValueError:
                continue
            stack[ind_previous][color_pattern == j] = stack[j][color_pattern == j]
            to_remove.append(j)
        clean_stack = np.delete(stack, to_remove, axis=0)
    else:
        clean_stack = stack.copy()

    assert len(np.where(np.nansum(clean_stack, axis=0) != raw_img)[0]) == 0

    return clean_stack


def put_together(R, G, B, G2, offsets):
    result = np.zeros((R.shape[0]*2, R.shape[1]*2))
    for colour, offset in zip([R,G,B,G2], offsets):
        x, y = offset
        result[x::2, y::2] = colour
    result = result.astype(R.dtype)
    return result


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


def cut_out_spectrum(raw_image):
    cut = raw_image[ymin:ymax, xmin:xmax]
    return cut
