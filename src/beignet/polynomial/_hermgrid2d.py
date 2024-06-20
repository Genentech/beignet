from .__grid import _grid
from ._evaluate_1d_physicists_hermite_series import (
    evaluate_1d_physicists_hermite_series,
)


def hermgrid2d(x, y, c):
    return _grid(evaluate_1d_physicists_hermite_series, c, x, y)
