from .__gridnd import _gridnd
from ._polyval import polyval


def polygrid3d(x, y, z, c):
    return _gridnd(polyval, c, x, y, z)
