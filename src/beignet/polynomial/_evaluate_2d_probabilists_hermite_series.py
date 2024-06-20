from .__evaluate import _evaluate
from ._evaluate_1d_probabilists_hermite_series import (
    evaluate_1d_probabilists_hermite_series,
)


def evaluate_2d_probabilists_hermite_series(x, y, c):
    return _evaluate(evaluate_1d_probabilists_hermite_series, c, x, y)
