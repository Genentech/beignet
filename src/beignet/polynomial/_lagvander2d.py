from .__vander_nd_flat import _vander_nd_flat
from ._lagvander import lagvander


def lagvander2d(x, y, deg):
    return _vander_nd_flat((lagvander, lagvander), (x, y), deg)
