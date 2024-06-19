from .__evaluate import _evaluate
from ._polyval import polyval


def polyval2d(x, y, c):
    return _evaluate(polyval, c, x, y)
