from .__grid import _grid
from ._evaluate_physicists_hermite_series_1d import (
    evaluate_physicists_hermite_series_1d,
)


def hermgrid3d(x, y, z, c):
    return _grid(evaluate_physicists_hermite_series_1d, c, x, y, z)
