from .__grid import _grid
from ._hermeval import hermeval


def hermegrid2d(x, y, c):
    return _grid(hermeval, c, x, y)
