from .__grid import _grid
from ._chebval import chebval


def chebgrid2d(x, y, c):
    return _grid(chebval, c, x, y)
