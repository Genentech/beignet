from .__evaluate import _evaluate
from ._chebval import chebval


def chebval2d(x, y, c):
    return _evaluate(chebval, c, x, y)
