from .__valnd import _valnd
from ._chebval import chebval


def chebval3d(x, y, z, c):
    return _valnd(chebval, c, x, y, z)
