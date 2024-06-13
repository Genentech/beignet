from .__vander_nd_flat import _vander_nd_flat
from ._lagvander import lagvander


def lagvander3d(x, y, z, deg):
    return _vander_nd_flat((lagvander, lagvander, lagvander), (x, y, z), deg)
