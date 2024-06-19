from .__vander_nd_flat import _vander_nd_flat
from ._hermvander import hermvander


def hermvander3d(x, y, z, deg):
    return _vander_nd_flat((hermvander, hermvander, hermvander), (x, y, z), deg)
