from .__evaluate import _evaluate
from ._evaluate_1d_physicists_hermite_series import (
    evaluate_1d_physicists_hermite_series,
)


def evaluate_3d_physicists_hermite_series(x, y, z, c):
    return _evaluate(evaluate_1d_physicists_hermite_series, c, x, y, z)
