from .__grid import _grid
from ._evaluate_laguerre_series_1d import evaluate_laguerre_series_1d


def evaluate_laguerre_series_grid_2d(x, y, c):
    return _grid(evaluate_laguerre_series_1d, c, x, y)
