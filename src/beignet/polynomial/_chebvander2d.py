from beignet.polynomial import _vander_nd_flat, chebvander


def chebvander2d(x, y, deg):
    return _vander_nd_flat((chebvander, chebvander), (x, y), deg)
