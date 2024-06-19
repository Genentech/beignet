from beignet.polynomial import _vander_nd_flat, hermvander


def hermvander3d(x, y, z, deg):
    return _vander_nd_flat((hermvander, hermvander, hermvander), (x, y, z), deg)
