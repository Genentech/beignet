from beignet.polynomial import legval
from beignet.polynomial.__valnd import _valnd


def legval2d(x, y, c):
    return _valnd(legval, c, x, y)
