from .__grid import _grid
from ._evaluate_legendre_series_1d import evaluate_legendre_series_1d


def leggrid3d(x, y, z, c):
    return _grid(evaluate_legendre_series_1d, c, x, y, z)
