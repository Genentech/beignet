import beignet.polynomial
import beignet.polynomial._evaluate_2d_legendre_series
import beignet.polynomial._legendre_series_vandermonde_2d
import numpy


def test_legvander2d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3))
    numpy.testing.assert_almost_equal(
        numpy.dot(
            beignet.polynomial._legvander2d.legendre_series_vandermonde_2d(
                x1, x2, [1, 2]
            ),
            c.flat,
        ),
        beignet.polynomial._legval2d.evaluate_2d_legendre_series(x1, x2, c),
    )

    numpy.testing.assert_(
        beignet.polynomial._legvander2d.legendre_series_vandermonde_2d(
            [x1], [x2], [1, 2]
        ).shape
        == (1, 5, 6)
    )
