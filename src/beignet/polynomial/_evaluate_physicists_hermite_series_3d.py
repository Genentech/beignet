from .__evaluate import _evaluate
from ._evaluate_physicists_hermite_series_1d import (
    evaluate_physicists_hermite_series_1d,
)


def evaluate_physicists_hermite_series_3d(x, y, z, c):
    return _evaluate(evaluate_physicists_hermite_series_1d, c, x, y, z)
