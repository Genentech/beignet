from .__gridnd import _gridnd
from ._chebval import chebval


def chebgrid2d(x, y, c):
    return _gridnd(chebval, c, x, y)
