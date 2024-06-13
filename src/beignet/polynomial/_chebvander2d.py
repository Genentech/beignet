from .__vander_nd_flat import _vander_nd_flat
from ._chebvander import chebvander


def chebvander2d(x, y, deg):
    return _vander_nd_flat((chebvander, chebvander), (x, y), deg)
