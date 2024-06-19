from .__evaluate import _evaluate
from ._lagval import lagval


def lagval3d(x, y, z, c):
    return _evaluate(lagval, c, x, y, z)
