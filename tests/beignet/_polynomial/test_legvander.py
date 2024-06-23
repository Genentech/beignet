import beignet.polynomial
import beignet.polynomial._evaluate_1d_legendre_series
import beignet.polynomial._legendre_series_vandermonde_1d
import numpy


def test_legvander():
    x = numpy.arange(3)
    v = beignet.polynomial._legvander.legendre_series_vandermonde_1d(x, 3)
    numpy.testing.assert_(v.shape == (3, 4))
    for i in range(4):
        coef = [0] * i + [1]
        numpy.testing.assert_almost_equal(
            v[..., i], beignet.polynomial._legval.evaluate_1d_legendre_series(x, coef)
        )

    x = numpy.array([[1, 2], [3, 4], [5, 6]])
    v = beignet.polynomial._legvander.legendre_series_vandermonde_1d(x, 3)
    numpy.testing.assert_(v.shape == (3, 2, 4))
    for i in range(4):
        coef = [0] * i + [1]
        numpy.testing.assert_almost_equal(
            v[..., i], beignet.polynomial._legval.evaluate_1d_legendre_series(x, coef)
        )

    numpy.testing.assert_raises(
        ValueError,
        beignet.polynomial._legvander.legendre_series_vandermonde_1d,
        (1, 2, 3),
        -1,
    )
