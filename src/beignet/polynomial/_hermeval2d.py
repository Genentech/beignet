from .__valnd import _valnd
from ._hermeval import hermeval


def hermeval2d(x, y, c):
    return _valnd(hermeval, c, x, y)
