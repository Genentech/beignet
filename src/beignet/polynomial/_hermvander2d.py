from .__vander_nd_flat import _vander_nd_flat
from ._hermvander import hermvander


def hermvander2d(x, y, deg):
    return _vander_nd_flat((hermvander, hermvander), (x, y), deg)
