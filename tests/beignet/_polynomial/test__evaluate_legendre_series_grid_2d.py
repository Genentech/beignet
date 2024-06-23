import beignet.polynomial
import numpy
import torch


def test_evaluate_legendre_series_grid_2d():
    c1d = numpy.array([2.0, 2.0, 2.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.polynomial.evaluate_power_series_1d(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    torch.testing.assert_close(
        beignet.polynomial.evaluate_legendre_series_grid_2d(
            x1,
            x2,
            c2d,
        ),
        torch.einsum(
            "i,j->ij",
            y1,
            y2,
        ),
    )

    z = torch.ones([2, 3])

    res = beignet.polynomial.evaluate_legendre_series_grid_2d(z, z, c2d)

    assert res.shape == (2, 3) * 2
