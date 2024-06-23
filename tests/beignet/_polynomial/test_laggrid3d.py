import beignet.polynomial
import numpy
import torch


def test_laggrid3d():
    c1d = numpy.array([9.0, -14.0, 6.0])

    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    x1, x2, x3 = x
    y1, y2, y3 = beignet.polynomial.evaluate_power_series_1d(x, [1.0, 2.0, 3.0])

    torch.testing.assert_close(
        beignet.polynomial.evaluate_laguerre_series_grid_3d(x1, x2, x3, c3d),
        numpy.einsum("i,j,k->ijk", y1, y2, y3),
    )

    z = numpy.ones((2, 3))
    assert (
        beignet.polynomial.evaluate_laguerre_series_grid_3d(z, z, z, c3d).shape
        == (2, 3) * 3
    )
