from .__valnd import _valnd
from ._lagval import lagval


def lagval3d(x, y, z, c):
    return _valnd(lagval, c, x, y, z)
