from .__valnd import _valnd
from ._lagval import lagval


def lagval2d(x, y, c):
    return _valnd(lagval, c, x, y)
