import beignet.polynomial
import numpy
import torch


def test_legendre_series_vandermonde_2d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3))

    torch.testing.assert_close(
        torch.dot(
            beignet.polynomial.legendre_series_vandermonde_2d(x1, x2, [1, 2]),
            torch.flatten(c),
        ),
        beignet.polynomial.evaluate_legendre_series_2d(x1, x2, c),
    )

    output = beignet.polynomial.legendre_series_vandermonde_2d([x1], [x2], [1, 2])
    assert output.shape == (1, 5, 6)
