from .__gridnd import _gridnd
from ._legval import legval


def leggrid3d(x, y, z, c):
    return _gridnd(legval, c, x, y, z)
