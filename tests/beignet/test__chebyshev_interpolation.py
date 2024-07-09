import beignet
import pytest
import torch


def test_chebyshev_interpolation():
    def f(x):
        return x * (x - 1) * (x - 2)

    with pytest.raises(ValueError):
        beignet.chebyshev_interpolation(f, -1)

    for i in range(1, 5):
        assert beignet.chebyshev_interpolation(f, i).shape == (i + 1,)

    def powx(x, p):
        return x**p

    x = torch.linspace(-1, 1, 10)

    for i in range(0, 10):
        for j in range(0, i + 1):
            c = beignet.chebyshev_interpolation(
                powx,
                i,
                j,
            )

            torch.testing.assert_close(
                beignet.evaluate_chebyshev_polynomial(x, c),
                powx(x, j),
            )
