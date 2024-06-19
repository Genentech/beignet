from beignet.polynomial import _vander_nd_flat, hermevander


def hermevander2d(x, y, deg):
    return _vander_nd_flat((hermevander, hermevander), (x, y), deg)
