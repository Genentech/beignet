from .__gridnd import _gridnd
from ._hermeval import hermeval


def hermegrid3d(x, y, z, c):
    return _gridnd(hermeval, c, x, y, z)
