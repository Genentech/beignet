from .__vander_nd_flat import _vander_nd_flat
from ._legvander import legvander


def legvander3d(x, y, z, deg):
    return _vander_nd_flat((legvander, legvander, legvander), (x, y, z), deg)
