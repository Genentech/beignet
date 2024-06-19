from .__grid import _grid
from ._legval import legval


def leggrid2d(x, y, c):
    return _grid(legval, c, x, y)
