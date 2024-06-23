import beignet.polynomial
import beignet.polynomial._chebgrid2d
import beignet.polynomial._evaluate_power_series_1d
import numpy


def test_chebgrid2d():
    c1d = numpy.array([2.5, 2.0, 1.5])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    x1, x2, x3 = x
    y1, y2, y3 = beignet.polynomial._polyval.evaluate_power_series_1d(
        x, [1.0, 2.0, 3.0]
    )

    numpy.testing.assert_almost_equal(
        beignet.polynomial._chebgrid2d.chebgrid2d(x1, x2, c2d),
        numpy.einsum("i,j->ij", y1, y2),
    )

    z = numpy.ones((2, 3))
    numpy.testing.assert_(
        beignet.polynomial._chebgrid2d.chebgrid2d(z, z, c2d).shape == (2, 3) * 2
    )
