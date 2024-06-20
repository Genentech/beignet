from .__grid import _grid
from ._evaluate_1d_laguerre_series import evaluate_1d_laguerre_series


def laggrid3d(x, y, z, c):
    return _grid(evaluate_1d_laguerre_series, c, x, y, z)
