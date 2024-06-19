from beignet.polynomial import _vander_nd_flat, legvander


def legvander3d(x, y, z, deg):
    return _vander_nd_flat((legvander, legvander, legvander), (x, y, z), deg)
