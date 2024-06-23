import beignet.polynomial
import beignet.polynomial._evaluate_probabilists_hermite_series_2d
import beignet.polynomial._probabilists_hermite_series_vandermonde_2d
import numpy
import torch


def test_probabilists_hermite_series_vandermonde_2d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3))
    van = beignet.polynomial.probabilists_hermite_series_vandermonde_2d(x1, x2, [1, 2])
    tgt = beignet.polynomial.evaluate_probabilists_hermite_series_2d(x1, x2, c)
    res = numpy.dot(van, c.flat)
    torch.testing.assert_close(res, tgt)

    van = beignet.polynomial.probabilists_hermite_series_vandermonde_2d(
        [x1], [x2], [1, 2]
    )
    assert van.shape == (1, 5, 6)
