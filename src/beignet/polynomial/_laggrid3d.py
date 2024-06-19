from .__grid import _grid
from ._lagval import lagval


def laggrid3d(x, y, z, c):
    return _grid(lagval, c, x, y, z)
