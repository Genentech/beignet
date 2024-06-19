from .__vander_nd_flat import _vander_nd_flat
from ._polyvander import polyvander


def polyvander3d(x, y, z, deg):
    return _vander_nd_flat((polyvander, polyvander, polyvander), (x, y, z), deg)
