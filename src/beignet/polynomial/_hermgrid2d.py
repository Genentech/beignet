from .__gridnd import _gridnd
from ._hermval import hermval


def hermgrid2d(x, y, c):
    return _gridnd(hermval, c, x, y)
