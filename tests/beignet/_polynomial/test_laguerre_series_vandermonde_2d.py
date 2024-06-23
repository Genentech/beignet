import beignet.polynomial
import numpy
import torch


def test_laguerre_series_vandermonde_2d():
    x1, x2, x3 = numpy.random.random((3, 5)) * 2 - 1
    c = numpy.random.random((2, 3))
    van = beignet.polynomial.laguerre_series_vandermonde_2d(x1, x2, [1, 2])
    tgt = beignet.polynomial.evaluate_laguerre_series_2d(x1, x2, c)
    res = numpy.dot(van, c.flat)
    torch.testing.assert_close(res, tgt)

    van = beignet.polynomial.laguerre_series_vandermonde_2d([x1], [x2], [1, 2])
    assert van.shape == (1, 5, 6)
