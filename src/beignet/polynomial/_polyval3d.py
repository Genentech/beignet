from .__valnd import _valnd
from ._polyval import polyval


def polyval3d(x, y, z, c):
    return _valnd(polyval, c, x, y, z)
