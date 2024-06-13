from .__gridnd import _gridnd
from ._lagval import lagval


def laggrid2d(x, y, c):
    return _gridnd(lagval, c, x, y)
