from .__valnd import _valnd
from ._hermval import hermval


def hermval2d(x, y, c):
    return _valnd(hermval, c, x, y)
