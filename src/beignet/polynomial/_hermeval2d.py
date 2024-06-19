from .__evaluate import _evaluate
from ._hermeval import hermeval


def hermeval2d(x, y, c):
    return _evaluate(hermeval, c, x, y)
