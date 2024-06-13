from .__gridnd import _gridnd
from ._hermval import hermval


def hermgrid3d(x, y, z, c):
    return _gridnd(hermval, c, x, y, z)
