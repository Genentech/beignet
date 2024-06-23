import beignet.polynomial
import numpy


def test_power_series_vandermonde_2d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1

    c = numpy.random.random((2, 3))

    numpy.testing.assert_almost_equal(
        numpy.dot(
            beignet.polynomial.power_series_vandermonde_2d(x1, x2, [1, 2]), c.flat
        ),
        beignet.polynomial.evaluate_power_series_2d(x1, x2, c),
    )

    numpy.testing.assert_(
        beignet.polynomial.power_series_vandermonde_2d([x1], [x2], [1, 2]).shape
        == (1, 5, 6)
    )
