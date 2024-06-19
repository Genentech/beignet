from beignet.polynomial import _vander_nd_flat, polyvander


def polyvander3d(x, y, z, deg):
    return _vander_nd_flat((polyvander, polyvander, polyvander), (x, y, z), deg)
