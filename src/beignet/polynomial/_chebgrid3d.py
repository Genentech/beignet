from .__gridnd import _gridnd
from ._chebval import chebval


def chebgrid3d(x, y, z, c):
    return _gridnd(chebval, c, x, y, z)
