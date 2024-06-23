import beignet.polynomial
import beignet.polynomial._chebgrid3d
import beignet.polynomial._evaluate_power_series_1d
import numpy


def test_chebgrid3d():
    c1d = numpy.array([2.5, 2.0, 1.5])

    c3d = numpy.einsum("i,j,k->ijk", c1d, c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    y = beignet.polynomial._polyval.evaluate_power_series_1d(x, [1.0, 2.0, 3.0])

    x1, x2, x3 = x
    y1, y2, y3 = y

    tgt = numpy.einsum("i,j,k->ijk", y1, y2, y3)
    numpy.testing.assert_almost_equal(
        beignet.polynomial._chebgrid3d.chebgrid3d(x1, x2, x3, c3d), tgt
    )

    z = numpy.ones((2, 3))
    numpy.testing.assert_(
        beignet.polynomial._chebgrid3d.chebgrid3d(z, z, z, c3d).shape == (2, 3) * 3
    )
