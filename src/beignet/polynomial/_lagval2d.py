from .__evaluate import _evaluate
from ._lagval import lagval


def lagval2d(x, y, c):
    return _evaluate(lagval, c, x, y)
