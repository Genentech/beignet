from .__valnd import _valnd
from ._chebval import chebval


def chebval2d(x, y, c):
    return _valnd(chebval, c, x, y)
