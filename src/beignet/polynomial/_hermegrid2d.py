from .__gridnd import _gridnd
from ._hermeval import hermeval


def hermegrid2d(x, y, c):
    return _gridnd(hermeval, c, x, y)
