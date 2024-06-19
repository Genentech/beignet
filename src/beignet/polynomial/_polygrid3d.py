from .__grid import _grid
from ._polyval import polyval


def polygrid3d(x, y, z, c):
    return _grid(polyval, c, x, y, z)
