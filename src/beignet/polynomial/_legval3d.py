from beignet.polynomial import legval
from beignet.polynomial.__valnd import _valnd


def legval3d(x, y, z, c):
    return _valnd(legval, c, x, y, z)
