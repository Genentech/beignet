from .__grid import _grid
from ._evaluate_laguerre_series_1d import evaluate_laguerre_series_1d


def laggrid3d(x, y, z, c):
    return _grid(evaluate_laguerre_series_1d, c, x, y, z)
