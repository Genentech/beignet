from beignet.polynomial import _vander_nd_flat, legvander


def legvander2d(x, y, deg):
    return _vander_nd_flat((legvander, legvander), (x, y), deg)
