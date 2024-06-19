from .__evaluate import _evaluate
from ._legval import legval


def legval3d(x, y, z, c):
    return _evaluate(legval, c, x, y, z)
