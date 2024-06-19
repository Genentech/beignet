from .__grid import _grid
from ._lagval import lagval


def laggrid2d(x, y, c):
    return _grid(lagval, c, x, y)
