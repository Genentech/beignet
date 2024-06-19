from .__evaluate import _evaluate
from ._hermval import hermval


def hermval2d(x, y, c):
    return _evaluate(hermval, c, x, y)
