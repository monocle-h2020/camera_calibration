import numpy as np


def pull_apart2(raw_img, color_pattern, color_desc="RGBG", remove=True):
    unique_colours = np.unique(color_pattern)

    stack = np.full((*unique_colours.shape, *raw_img.shape), np.nan)
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
