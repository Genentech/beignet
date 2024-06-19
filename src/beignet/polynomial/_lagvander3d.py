from beignet.polynomial import _vander_nd_flat, lagvander


def lagvander3d(x, y, z, deg):
    return _vander_nd_flat((lagvander, lagvander, lagvander), (x, y, z), deg)
