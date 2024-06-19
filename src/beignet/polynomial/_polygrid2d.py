from .__grid import _grid
from ._polyval import polyval


def polygrid2d(x, y, c):
    return _grid(polyval, c, x, y)
