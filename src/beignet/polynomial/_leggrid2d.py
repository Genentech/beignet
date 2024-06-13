from .__gridnd import _gridnd
from ._legval import legval


def leggrid2d(x, y, c):
    return _gridnd(legval, c, x, y)
