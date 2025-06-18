
def colormap_value(v, vmin, vmax):
    if v < vmin: v = vmin
    if v > vmax: v = vmax

    r, g, b = 1., 1., 1.

    dv = vmax - vmin

    if v < (vmin + .25 * dv):
        r = 0.
        g = 4 * (v - vmin) / dv
    elif v < (vmin + .5 * dv):
        r = 0.
        b = 1 + 4 * (vmin + .25 * dv - v) / dv
    elif v < (vmin + .75 * dv):
        r = 4 * (v - vmin - .5 * dv) / dv
        b = 0
    else:
        g = 1 + 4 * (vmin + .75 * dv - v) / dv
        b = 0

    return r, g, b