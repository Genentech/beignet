from .__valnd import _valnd
from ._hermval import hermval


def hermval3d(x, y, z, c):
    return _valnd(hermval, c, x, y, z)
