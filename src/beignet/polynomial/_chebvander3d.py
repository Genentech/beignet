from .__vander_nd_flat import _vander_nd_flat
from ._chebvander import chebvander


def chebvander3d(x, y, z, deg):
    return _vander_nd_flat((chebvander, chebvander, chebvander), (x, y, z), deg)
