import beignet.polynomial
import numpy
import pytest
import torch


def test_chebyshev_interpolation():
    def f(x):
        return x * (x - 1) * (x - 2)

    with pytest.raises(ValueError):
        beignet.polynomial.chebyshev_interpolation(f, -1)

    with pytest.raises(TypeError):
        beignet.polynomial.chebyshev_interpolation(f, 10.0)

    for degree in range(1, 5):
        assert beignet.polynomial.chebyshev_interpolation(f, degree).shape == (
            degree + 1,
        )

    def powx(x, p):
        return x**p

    x = numpy.linspace(-1, 1, 10)

    for degree in range(0, 10):
        for p in range(0, degree + 1):
            c = beignet.polynomial.chebyshev_interpolation(powx, degree, (p,))
            torch.testing.assert_close(
                beignet.polynomial.evaluate_chebyshev_series_1d(x, c),
                powx(x, p),
                decimal=12,
            )
