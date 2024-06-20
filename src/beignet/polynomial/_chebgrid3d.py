from .__grid import _grid
from ._evaluate_1d_chebyshev_series import evaluate_1d_chebyshev_series


def chebgrid3d(x, y, z, c):
    return _grid(evaluate_1d_chebyshev_series, c, x, y, z)
