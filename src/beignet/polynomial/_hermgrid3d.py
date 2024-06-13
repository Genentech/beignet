from beignet._evaluate_physicists_hermite_series import (
    evaluate_physicists_hermite_series,
)

from .__gridnd import _gridnd


def hermgrid3d(x, y, z, c):
    return _gridnd(evaluate_physicists_hermite_series, c, x, y, z)
