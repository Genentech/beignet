from .__evaluate import _evaluate
from ._legval import legval


def legval2d(x, y, c):
    return _evaluate(legval, c, x, y)
