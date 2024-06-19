from .__evaluate import _evaluate
from ._chebval import chebval


def chebval3d(x, y, z, c):
    return _evaluate(chebval, c, x, y, z)
