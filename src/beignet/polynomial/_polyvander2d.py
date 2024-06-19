from beignet.polynomial import _vander_nd_flat, polyvander


def polyvander2d(x, y, deg):
    return _vander_nd_flat((polyvander, polyvander), (x, y), deg)
