from .__grid import _grid
from ._hermval import hermval


def hermgrid2d(x, y, c):
    return _grid(hermval, c, x, y)
