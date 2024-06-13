from .__vander_nd_flat import _vander_nd_flat
from ._hermevander import hermevander


def hermevander2d(x, y, deg):
    return _vander_nd_flat((hermevander, hermevander), (x, y), deg)
