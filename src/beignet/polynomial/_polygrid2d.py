from .__gridnd import _gridnd
from ._polyval import polyval


def polygrid2d(x, y, c):
    return _gridnd(polyval, c, x, y)
