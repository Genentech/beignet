import beignet.polynomial
import numpy
import torch


def test_legendre_series_vandermonde_3d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3, 4))
    torch.testing.assert_close(
        numpy.dot(
            beignet.polynomial.legendre_series_vandermonde_3d(x1, x2, x3, [1, 2, 3]),
            c.flat,
        ),
        beignet.polynomial.evaluate_legendre_series_3d(x1, x2, x3, c),
    )

    numpy.testing.assert_(
        beignet.polynomial.legendre_series_vandermonde_3d(
            [x1], [x2], [x3], [1, 2, 3]
        ).shape
        == (1, 5, 24)
    )
