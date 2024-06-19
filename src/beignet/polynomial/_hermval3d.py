from .__evaluate import _evaluate
from ._hermval import hermval


def hermval3d(x, y, z, c):
    return _evaluate(hermval, c, x, y, z)
