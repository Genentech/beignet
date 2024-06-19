from .__grid import _grid
from ._hermeval import hermeval


def hermegrid3d(x, y, z, c):
    return _grid(hermeval, c, x, y, z)
