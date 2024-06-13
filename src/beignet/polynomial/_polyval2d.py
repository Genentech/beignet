from .__valnd import _valnd
from ._polyval import polyval


def polyval2d(x, y, c):
    return _valnd(polyval, c, x, y)
