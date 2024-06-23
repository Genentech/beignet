from .__grid import _grid
from ._evaluate_power_series_1d import evaluate_power_series_1d


def polygrid2d(x, y, c):
    return _grid(evaluate_power_series_1d, c, x, y)
