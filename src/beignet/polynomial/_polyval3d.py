from .__evaluate import _evaluate
from ._polyval import polyval


def polyval3d(x, y, z, c):
    return _evaluate(polyval, c, x, y, z)
