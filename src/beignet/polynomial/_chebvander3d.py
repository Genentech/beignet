from beignet.polynomial import _vander_nd_flat, chebvander


def chebvander3d(x, y, z, deg):
    return _vander_nd_flat((chebvander, chebvander, chebvander), (x, y, z), deg)
