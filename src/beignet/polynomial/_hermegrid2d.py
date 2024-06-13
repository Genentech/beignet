from beignet._evaluate_probabilists_hermite_series import (
    evaluate_probabilists_hermite_series,
)

from .__gridnd import _gridnd


def hermegrid2d(x, y, c):
    return _gridnd(evaluate_probabilists_hermite_series, c, x, y)
