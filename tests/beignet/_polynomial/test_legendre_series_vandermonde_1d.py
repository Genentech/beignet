import beignet.polynomial
import beignet.polynomial._evaluate_legendre_series_1d
import beignet.polynomial._legendre_series_vandermonde_1d
import numpy


def test_legendre_series_vandermonde_1d():
    x = numpy.arange(3)
    v = beignet.polynomial.legendre_series_vandermonde_1d(x, 3)
    numpy.testing.assert_(v.shape == (3, 4))
    for i in range(4):
        coef = [0] * i + [1]
        numpy.testing.assert_almost_equal(
            v[..., i], beignet.polynomial.evaluate_legendre_series_1d(x, coef)
        )

    x = numpy.array([[1, 2], [3, 4], [5, 6]])
    v = beignet.polynomial.legendre_series_vandermonde_1d(x, 3)
    numpy.testing.assert_(v.shape == (3, 2, 4))
    for i in range(4):
        coef = [0] * i + [1]
        numpy.testing.assert_almost_equal(
            v[..., i], beignet.polynomial.evaluate_legendre_series_1d(x, coef)
        )

    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial.legendre_series_vandermonde_1d,
        (1, 2, 3),
        -1,
    )
