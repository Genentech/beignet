from beignet._evaluate_physicists_hermite_series import (
    evaluate_physicists_hermite_series,
)

from .__gridnd import _gridnd


def hermgrid2d(x, y, c):
    return _gridnd(evaluate_physicists_hermite_series, c, x, y)
