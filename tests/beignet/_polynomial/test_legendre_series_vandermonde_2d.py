import beignet.polynomial
import beignet.polynomial._evaluate_legendre_series_2d
import beignet.polynomial._legendre_series_vandermonde_2d
import numpy
import torch


def test_legendre_series_vandermonde_2d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3))
    torch.testing.assert_close(
        numpy.dot(
            beignet.polynomial.legendre_series_vandermonde_2d(x1, x2, [1, 2]),
            c.flat,
        ),
        beignet.polynomial.evaluate_legendre_series_2d(x1, x2, c),
    )

    numpy.testing.assert_(
        beignet.polynomial.legendre_series_vandermonde_2d([x1], [x2], [1, 2]).shape
        == (1, 5, 6)
    )
