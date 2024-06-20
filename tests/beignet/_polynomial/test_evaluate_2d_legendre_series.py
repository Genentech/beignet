import beignet.polynomial
import beignet.polynomial._evaluate_1d_power_series
import beignet.polynomial._evaluate_2d_legendre_series
import numpy


def test_evaluate_2d_legendre_series():
    c1d = numpy.array([2.0, 2.0, 2.0])
    c2d = numpy.einsum("i,j->ij", c1d, c1d)

    x = numpy.random.random((3, 5)) * 2 - 1
    x1, x2, x3 = x
    y1, y2, y3 = beignet.polynomial._polyval.evaluate_1d_power_series(
        x, [1.0, 2.0, 3.0]
    )

    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial._legval2d.evaluate_2d_legendre_series,
        x1,
        x2[:2],
        c2d,
    )

    tgt = y1 * y2
    numpy.testing.assert_almost_equal(
        beignet.polynomial._legval2d.evaluate_2d_legendre_series(x1, x2, c2d), tgt
    )

    z = numpy.ones((2, 3))
    numpy.testing.assert_(
        beignet.polynomial._legval2d.evaluate_2d_legendre_series(z, z, c2d).shape
        == (2, 3)
    )
