from .__grid import _grid
from ._evaluate_1d_power_series import evaluate_1d_power_series


def polygrid2d(x, y, c):
    return _grid(evaluate_1d_power_series, c, x, y)
