from .__vander_nd_flat import _vander_nd_flat
from ._hermevander import hermevander


def hermevander3d(x, y, z, deg):
    return _vander_nd_flat((hermevander, hermevander, hermevander), (x, y, z), deg)
