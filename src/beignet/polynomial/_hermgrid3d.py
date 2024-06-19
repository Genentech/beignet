from .__grid import _grid
from ._hermval import hermval


def hermgrid3d(x, y, z, c):
    return _grid(hermval, c, x, y, z)
