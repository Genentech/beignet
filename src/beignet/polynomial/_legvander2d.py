from .__vander_nd_flat import _vander_nd_flat
from ._legvander import legvander


def legvander2d(x, y, deg):
    return _vander_nd_flat((legvander, legvander), (x, y), deg)
