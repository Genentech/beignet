from .__grid import _grid
from ._evaluate_1d_legendre_series import evaluate_1d_legendre_series


def leggrid3d(x, y, z, c):
    return _grid(evaluate_1d_legendre_series, c, x, y, z)
