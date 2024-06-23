from .__grid import _grid
from ._evaluate_probabilists_hermite_series_1d import (
    evaluate_probabilists_hermite_series_1d,
)


def hermegrid3d(x, y, z, c):
    return _grid(evaluate_probabilists_hermite_series_1d, c, x, y, z)
