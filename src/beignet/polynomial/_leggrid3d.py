from beignet._evaluate_legendre_series import evaluate_legendre_series

from .__gridnd import _gridnd


def leggrid3d(x, y, z, c):
    return _gridnd(evaluate_legendre_series, c, x, y, z)
