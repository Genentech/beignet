from beignet.polynomial import _vander_nd_flat, hermevander


def hermevander3d(x, y, z, deg):
    return _vander_nd_flat((hermevander, hermevander, hermevander), (x, y, z), deg)
