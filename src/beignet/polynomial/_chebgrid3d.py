from .__grid import _grid
from ._chebval import chebval


def chebgrid3d(x, y, z, c):
    return _grid(chebval, c, x, y, z)
