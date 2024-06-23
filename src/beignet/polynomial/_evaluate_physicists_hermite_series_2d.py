from .__evaluate import _evaluate
from ._evaluate_physicists_hermite_series_1d import (
    evaluate_physicists_hermite_series_1d,
)


def evaluate_physicists_hermite_series_2d(x, y, c):
    return _evaluate(evaluate_physicists_hermite_series_1d, c, x, y)
