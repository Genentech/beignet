from .__vander_nd_flat import _vander_nd_flat
from ._polyvander import polyvander


def polyvander2d(x, y, deg):
    return _vander_nd_flat((polyvander, polyvander), (x, y), deg)
