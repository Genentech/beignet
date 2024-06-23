import beignet.polynomial
import numpy
import torch


def test_laggrid2d():
    c1d = numpy.array([9.0, -14.0, 6.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    x1, x2, x3 = x
    y1, y2, y3 = beignet.polynomial.evaluate_power_series_1d(x, [1.0, 2.0, 3.0])

    torch.testing.assert_close(
        beignet.polynomial._laggrid2d.laggrid2d(x1, x2, c2d),
        numpy.einsum("i,j->ij", y1, y2),
    )

    z = numpy.ones((2, 3))
    numpy.testing.assert_(
        beignet.polynomial._laggrid2d.laggrid2d(z, z, c2d).shape == (2, 3) * 2
    )
