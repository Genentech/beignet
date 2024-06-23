from .__grid import _grid
from ._evaluate_power_series_1d import evaluate_power_series_1d


def evaluate_power_series_grid_3d(x, y, z, c):
    return _grid(evaluate_power_series_1d, c, x, y, z)
