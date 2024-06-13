from .__valnd import _valnd
from ._hermeval import hermeval


def hermeval3d(x, y, z, c):
    return _valnd(hermeval, c, x, y, z)
