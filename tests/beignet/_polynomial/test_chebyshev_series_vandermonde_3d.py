import beignet.polynomial
import beignet.polynomial._chebyshev_series_vandermonde_3d
import beignet.polynomial._evaluate_chebyshev_series_3d
import numpy


def test_chebyshev_series_vandermonde_3d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3, 4))
    numpy.testing.assert_almost_equal(
        numpy.dot(
            beignet.polynomial.chebyshev_series_vandermonde_3d(x1, x2, x3, [1, 2, 3]),
            c.flat,
        ),
        beignet.polynomial.evaluate_chebyshev_series_3d(x1, x2, x3, c),
    )
    numpy.testing.assert_(
        beignet.polynomial.chebyshev_series_vandermonde_3d(
            [x1], [x2], [x3], [1, 2, 3]
        ).shape
        == (1, 5, 24)
    )
