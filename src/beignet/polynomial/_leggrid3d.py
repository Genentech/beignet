from .__grid import _grid
from ._legval import legval


def leggrid3d(x, y, z, c):
    return _grid(legval, c, x, y, z)
