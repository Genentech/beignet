from beignet.polynomial import _vander_nd_flat, hermvander


def hermvander2d(x, y, deg):
    return _vander_nd_flat((hermvander, hermvander), (x, y), deg)
