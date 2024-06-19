from .__evaluate import _evaluate
from ._hermeval import hermeval


def hermeval3d(x, y, z, c):
    return _evaluate(hermeval, c, x, y, z)
